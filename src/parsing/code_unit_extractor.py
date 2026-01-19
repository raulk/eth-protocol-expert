"""Code Unit Extractor - Extract functions, structs, interfaces from parsed files."""

from dataclasses import dataclass
from enum import Enum

import structlog

from src.parsing.treesitter_parser import ParsedFile

logger = structlog.get_logger()


class UnitType(Enum):
    """Types of code units that can be extracted."""

    FUNCTION = "function"
    STRUCT = "struct"
    INTERFACE = "interface"
    METHOD = "method"
    CONSTANT = "constant"


@dataclass
class CodeUnit:
    """A discrete unit of code extracted from a source file."""

    name: str
    unit_type: UnitType
    content: str
    file_path: str
    line_range: tuple[int, int]
    dependencies: list[str]


class CodeUnitExtractor:
    """Extract discrete code units from parsed source files."""

    def extract_all(self, parsed: ParsedFile) -> list[CodeUnit]:
        """Extract all code units from a parsed file."""
        units: list[CodeUnit] = []

        units.extend(self.extract_functions(parsed))
        units.extend(self.extract_types(parsed))

        logger.debug(
            "extracted_code_units",
            path=parsed.path,
            count=len(units),
        )

        return units

    def extract_functions(self, parsed: ParsedFile) -> list[CodeUnit]:
        """Extract all functions and methods from a parsed file."""
        units: list[CodeUnit] = []

        for func in parsed.functions:
            content_parts = []

            if func.docstring:
                if parsed.language == "go":
                    for line in func.docstring.split("\n"):
                        content_parts.append(f"// {line}")
                elif parsed.language == "rust":
                    for line in func.docstring.split("\n"):
                        content_parts.append(f"/// {line}")

            if parsed.language == "go":
                content_parts.append(f"func {func.signature} {func.body}")
            elif parsed.language == "rust":
                content_parts.append(f"fn {func.signature} {func.body}")
            else:
                content_parts.append(f"{func.signature} {func.body}")

            content = "\n".join(content_parts)

            is_method = (
                "(" in func.signature.split(func.name)[0] if func.name in func.signature else False
            )
            unit_type = UnitType.METHOD if is_method else UnitType.FUNCTION

            deps = self._extract_dependencies_from_body(func.body, parsed.imports)

            units.append(
                CodeUnit(
                    name=func.name,
                    unit_type=unit_type,
                    content=content,
                    file_path=parsed.path,
                    line_range=(func.start_line, func.end_line),
                    dependencies=deps,
                )
            )

        return units

    def extract_types(self, parsed: ParsedFile) -> list[CodeUnit]:
        """Extract all struct/interface definitions from a parsed file."""
        units: list[CodeUnit] = []

        for struct in parsed.structs:
            if parsed.language == "go":
                fields_str = "\n".join(f"    {name} {typ}" for name, typ in struct.fields)
                content = f"type {struct.name} struct {{\n{fields_str}\n}}"
            elif parsed.language == "rust":
                fields_str = "\n".join(f"    {name}: {typ}," for name, typ in struct.fields)
                content = f"struct {struct.name} {{\n{fields_str}\n}}"
            else:
                content = f"struct {struct.name}"

            deps = []
            for _, field_type in struct.fields:
                dep_type = self._extract_type_dependency(field_type)
                if dep_type and dep_type not in deps:
                    deps.append(dep_type)

            units.append(
                CodeUnit(
                    name=struct.name,
                    unit_type=UnitType.STRUCT,
                    content=content,
                    file_path=parsed.path,
                    line_range=(struct.start_line, struct.end_line),
                    dependencies=deps,
                )
            )

        return units

    def find_references(
        self,
        unit: CodeUnit,
        codebase: list[ParsedFile],
    ) -> list[str]:
        """Find all files that reference a given code unit."""
        references: list[str] = []

        for parsed in codebase:
            if parsed.path == unit.file_path:
                continue

            for func in parsed.functions:
                if unit.name in func.body:
                    if parsed.path not in references:
                        references.append(parsed.path)
                    break

            for struct in parsed.structs:
                for _, field_type in struct.fields:
                    if unit.name in field_type:
                        if parsed.path not in references:
                            references.append(parsed.path)
                        break

        logger.debug(
            "found_references",
            unit=unit.name,
            count=len(references),
        )

        return references

    def _extract_dependencies_from_body(
        self,
        body: str,
        imports: list[str],
    ) -> list[str]:
        deps: list[str] = []

        for imp in imports:
            package_name = imp.split("/")[-1] if "/" in imp else imp
            package_name = package_name.split("::")[-1] if "::" in package_name else package_name

            if f"{package_name}." in body or f"{package_name}::" in body:
                if imp not in deps:
                    deps.append(imp)

        return deps

    def _extract_type_dependency(self, field_type: str) -> str | None:
        cleaned = field_type.strip()

        for prefix in ("*", "&", "[]", "Option<", "Vec<", "Box<", "Arc<", "Rc<"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]

        for suffix in (">",):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]

        primitives = {
            "string",
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float32",
            "float64",
            "bool",
            "byte",
            "rune",
            "error",
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
            "f32",
            "f64",
            "str",
            "String",
            "()",
        }

        if cleaned.lower() in {p.lower() for p in primitives}:
            return None

        return cleaned if cleaned else None
