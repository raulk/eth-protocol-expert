"""GitHub Issues Loader - Fetch issues and PRs for ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

DEFAULT_REPOS: list[str] = [
    "ethereum/EIPs",
    "ethereum/ERCs",
    "ethereum/RIPs",
]


@dataclass
class GitHubComment:
    """A GitHub issue or PR comment."""

    author: str | None
    body: str
    created_at: str | None
    updated_at: str | None
    url: str | None


@dataclass
class GitHubIssueDocument:
    """Normalized GitHub issue/PR document for ingestion."""

    document_id: str
    document_type: str  # github_issue | github_pr
    source: str
    title: str
    content: str
    metadata: dict[str, Any]
    updated_at: str | None
    fetched_at: str


class GitHubIssuesLoader:
    """Load GitHub issues/PRs via REST API."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        timeout: float = 30.0,
        per_page: int = 100,
    ) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not set")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.per_page = min(per_page, 100)

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "eth-protocol-expert",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def fetch_issues(
        self,
        owner: str,
        repo: str,
        *,
        since: str | None = None,
        state: str = "all",
        labels: list[str] | None = None,
        include_issues: bool = True,
        include_prs: bool = True,
        include_comments: bool = True,
        max_comments: int | None = 50,
        max_items: int | None = None,
        max_pages: int | None = None,
    ) -> tuple[list[GitHubIssueDocument], str | None]:
        """Fetch issues and PRs, optionally filtered by updated time.

        Returns:
            (documents, latest_updated_at)
        """
        documents: list[GitHubIssueDocument] = []
        latest_dt: datetime | None = None
        latest_str: str | None = None

        params = {
            "state": state,
            "per_page": self.per_page,
        }
        if since:
            params["since"] = since
        if labels:
            params["labels"] = ",".join(labels)

        url = f"{self.base_url}/repos/{owner}/{repo}/issues"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            page = 1
            while True:
                if max_pages and page > max_pages:
                    break

                params["page"] = page
                response = await client.get(url, headers=self._headers(), params=params)

                if response.status_code == 403:
                    logger.error(
                        "github_rate_limited",
                        owner=owner,
                        repo=repo,
                        remaining=response.headers.get("X-RateLimit-Remaining"),
                    )
                    break

                response.raise_for_status()
                items = response.json()
                if not items:
                    break

                for item in items:
                    is_pr = bool(item.get("pull_request"))
                    if is_pr and not include_prs:
                        continue
                    if (not is_pr) and not include_issues:
                        continue

                    comments: list[GitHubComment] = []
                    if include_comments and item.get("comments", 0) > 0:
                        comments = await self._fetch_comments(
                            client,
                            owner=owner,
                            repo=repo,
                            issue_number=item.get("number"),
                            max_comments=max_comments,
                        )

                    doc = self._to_document(owner, repo, item, comments)
                    documents.append(doc)

                    updated_at = item.get("updated_at")
                    updated_dt = self._parse_timestamp(updated_at)
                    if updated_dt and (latest_dt is None or updated_dt > latest_dt):
                        latest_dt = updated_dt
                        latest_str = updated_at

                    if max_items and len(documents) >= max_items:
                        break

                if max_items and len(documents) >= max_items:
                    break

                page += 1

        logger.info(
            "github_issues_loaded",
            owner=owner,
            repo=repo,
            count=len(documents),
            latest=latest_str,
        )
        return documents, latest_str

    async def _fetch_comments(
        self,
        client: httpx.AsyncClient,
        *,
        owner: str,
        repo: str,
        issue_number: int,
        max_comments: int | None = None,
    ) -> list[GitHubComment]:
        """Fetch issue/PR comments (paginated)."""
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
        comments: list[GitHubComment] = []
        page = 1

        while True:
            params = {"per_page": self.per_page, "page": page}
            response = await client.get(url, headers=self._headers(), params=params)
            if response.status_code == 403:
                logger.warning(
                    "github_comment_rate_limited",
                    owner=owner,
                    repo=repo,
                    issue_number=issue_number,
                )
                break

            response.raise_for_status()
            items = response.json()
            if not items:
                break

            for item in items:
                comments.append(
                    GitHubComment(
                        author=(item.get("user") or {}).get("login"),
                        body=item.get("body") or "",
                        created_at=item.get("created_at"),
                        updated_at=item.get("updated_at"),
                        url=item.get("html_url"),
                    )
                )
                if max_comments and len(comments) >= max_comments:
                    return comments

            page += 1

        return comments

    def _to_document(
        self,
        owner: str,
        repo: str,
        item: dict[str, Any],
        comments: list[GitHubComment],
    ) -> GitHubIssueDocument:
        is_pr = bool(item.get("pull_request"))
        number = item.get("number")
        doc_type = "github_pr" if is_pr else "github_issue"
        source = f"github/{owner}/{repo}"
        document_id = f"github-{owner}-{repo}-{'pr' if is_pr else 'issue'}-{number}"

        title = item.get("title") or f"{doc_type} #{number}"
        body = item.get("body") or ""
        label_names = [label.get("name") for label in item.get("labels", []) if label]

        content = self._format_issue_text(
            owner=owner,
            repo=repo,
            title=title,
            number=number,
            state=item.get("state"),
            author=(item.get("user") or {}).get("login"),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            labels=label_names,
            url=item.get("html_url"),
            body=body,
            comments=comments,
        )

        metadata = {
            "repo": f"{owner}/{repo}",
            "number": number,
            "state": item.get("state"),
            "labels": label_names,
            "author": (item.get("user") or {}).get("login"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "closed_at": item.get("closed_at"),
            "is_pr": is_pr,
            "comments": item.get("comments", 0),
            "url": item.get("html_url"),
        }

        return GitHubIssueDocument(
            document_id=document_id,
            document_type=doc_type,
            source=source,
            title=title,
            content=content,
            metadata=metadata,
            updated_at=item.get("updated_at"),
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    def _format_issue_text(
        self,
        *,
        owner: str,
        repo: str,
        title: str,
        number: int,
        state: str | None,
        author: str | None,
        created_at: str | None,
        updated_at: str | None,
        labels: list[str],
        url: str | None,
        body: str,
        comments: list[GitHubComment],
    ) -> str:
        lines = [
            f"# {title}",
            "",
            f"**Repo**: {owner}/{repo}",
            f"**Number**: {number}",
            f"**State**: {state or 'unknown'}",
            f"**Author**: {author or 'unknown'}",
            f"**Created**: {created_at or 'unknown'}",
            f"**Updated**: {updated_at or 'unknown'}",
            f"**Labels**: {', '.join(labels) if labels else 'none'}",
            f"**URL**: {url or 'unknown'}",
            "",
            "## Body",
            body.strip(),
        ]

        if comments:
            lines.append("")
            lines.append("## Comments")
            for comment in comments:
                lines.append("")
                header = f"### {comment.author or 'unknown'} ({comment.created_at or 'unknown'})"
                lines.append(header)
                if comment.body:
                    lines.append(comment.body.strip())

        return "\n".join(lines).strip()

    def _parse_timestamp(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        except ValueError:
            return None
