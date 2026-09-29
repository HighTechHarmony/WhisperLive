"""PipeWire port discovery and linking for the capture node."""

from __future__ import annotations

import subprocess
import os


class PipeWireError(RuntimeError):
    """Raised when PipeWire cannot be queried or linked."""


def _spa_prop_value(value: str) -> str:
    return str(value).replace('"', "").replace("\\", "").strip()


def configure_pipewire_node(node_name: str, node_description: str) -> None:
    """Set stable ALSA/PipeWire properties before PortAudio creates its node."""
    if not os.environ.get("PIPEWIRE_ALSA"):
        os.environ["PIPEWIRE_ALSA"] = (
            "{ "
            f'node.name = "{_spa_prop_value(node_name)}" '
            f'node.description = "{_spa_prop_value(node_description)}" '
            'application.name = "WhisperLive" '
            "}"
        )
    os.environ.setdefault(
        "PIPEWIRE_PROPS",
        '{ node.description = "%s" }' % _spa_prop_value(node_description),
    )


class PipeWireRouter:
    def __init__(self, node_name: str, sources: list[str], runner=None):
        self.node_name = node_name
        self.sources = sources
        self._runner = runner or subprocess.run

    def _ports(self, direction: str) -> list[str]:
        try:
            result = self._runner(
                ["pw-link", direction],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PipeWireError(f"Could not query PipeWire ports: {exc}") from exc
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def input_ports(self) -> list[str]:
        return [port for port in self._ports("-i") if self.node_name in port]

    def output_ports(self) -> list[str]:
        return self._ports("-o")

    @staticmethod
    def _match_source(source: str, ports: list[str]) -> str | None:
        source_lower = source.lower()
        for port in ports:
            if port.lower() == source_lower or port.rsplit(":", 1)[-1].lower() == source_lower:
                return port
        for port in ports:
            if source_lower in port.lower():
                return port
        return None

    def relink(self) -> tuple[list[str], list[str]]:
        """Link configured source ports to every input port on this node.

        Returns ``(linked_ports, missing_sources)`` so the GUI can report partial
        routing without treating a disconnected optional source as fatal.
        """
        input_ports = self.input_ports()
        output_ports = self.output_ports()
        if not input_ports:
            raise PipeWireError(f"No input ports found for node {self.node_name!r}")

        linked = []
        missing = []
        for source in self.sources:
            source_port = self._match_source(source, output_ports)
            if source_port is None:
                missing.append(source)
                continue
            for input_port in input_ports:
                try:
                    self._runner(
                        ["pw-link", source_port, input_port],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError) as exc:
                    if isinstance(exc, subprocess.CalledProcessError):
                        message = f"{exc.stdout or ''}\n{exc.stderr or ''}"
                        if "File exists" in message:
                            continue
                    raise PipeWireError(
                        f"Could not link {source_port} to {input_port}: {exc}"
                    ) from exc
            linked.append(source_port)
        return linked, missing