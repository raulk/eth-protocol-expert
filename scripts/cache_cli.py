#!/usr/bin/env python3
"""Cache management CLI.

Usage:
    uv run python scripts/cache_cli.py stats
    uv run python scripts/cache_cli.py stats --source ethresearch
    uv run python scripts/cache_cli.py list --source ethresearch
    uv run python scripts/cache_cli.py list --source ethresearch --limit 20
"""

import argparse

from src.ingestion.cache import RawContentCache

SOURCES = ["arxiv", "ethresearch", "magicians"]


def cmd_stats(args: argparse.Namespace) -> None:
    """Show cache statistics."""
    cache = RawContentCache()
    sources = [args.source] if args.source else SOURCES

    total_entries = 0
    total_mb = 0.0

    for source in sources:
        stats = cache.stats(source)
        if stats["entry_count"] > 0 or args.source:
            print(f"\n{source}:")
            print(f"  Entries: {stats['entry_count']}")
            print(f"  Size: {stats['total_mb']} MB")
            print(f"  Last sync: {stats['last_sync'] or 'never'}")
            total_entries += stats["entry_count"]
            total_mb += stats["total_mb"]

    if not args.source:
        print(f"\nTotal: {total_entries} entries, {round(total_mb, 2)} MB")


def cmd_list(args: argparse.Namespace) -> None:
    """List cached entries."""
    cache = RawContentCache()
    entries = cache.list_entries(args.source)

    if not entries:
        print(f"No entries cached for {args.source}")
        return

    print(f"Cached entries for {args.source} ({len(entries)} total):\n")

    for entry in entries[: args.limit]:
        title = entry.meta.get("title", "N/A")
        if len(title) > 60:
            title = title[:57] + "..."
        print(f"  {entry.item_id}: {title}")
        print(f"    Fetched: {entry.fetched_at}")
        print(f"    Size: {entry.content_bytes:,} bytes")
        print()


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear cache entries for a source."""
    cache = RawContentCache()

    if not args.force:
        stats = cache.stats(args.source)
        print(
            f"About to clear {stats['entry_count']} entries ({stats['total_mb']} MB) for {args.source}"
        )
        confirm = input("Are you sure? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    count = cache.clear_source(args.source)
    print(f"Cleared {count} entries for {args.source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.add_argument(
        "--source",
        choices=SOURCES,
        help="Filter by source",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List cached entries")
    list_parser.add_argument(
        "--source",
        required=True,
        choices=SOURCES,
        help="Source to list",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max entries to show (default: 50)",
    )

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache for a source")
    clear_parser.add_argument(
        "--source",
        required=True,
        choices=SOURCES,
        help="Source to clear",
    )
    clear_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation",
    )

    args = parser.parse_args()

    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "clear":
        cmd_clear(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
