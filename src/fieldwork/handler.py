"""
Spatial Atlas — FieldWorkArena Domain Handler

Main entry point for FieldWorkArena tasks. Orchestrates:
1. Goal parsing (structured task extraction)
2. Multimodal file processing (images, PDFs, videos, text)
3. Spatial scene graph construction
4. Entropy-guided reasoning
5. Output format matching

Receives: text goal + file attachments from green agent
Returns: formatted answer text
"""

import logging

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from config import Config
from llm import LLMClient
from fieldwork.parser import GoalParser
from fieldwork.vision import VisionPipeline
from fieldwork.spatial import SpatialAnalyzer
from fieldwork.reasoner import FieldWorkReasoner
from fieldwork.formatter import AnswerFormatter

logger = logging.getLogger("spatial-atlas.fieldwork")


class FieldWorkHandler:
    """Handle FieldWorkArena benchmark tasks."""

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.parser = GoalParser()
        self.vision = VisionPipeline(llm, max_video_frames=config.max_video_frames)
        self.spatial = SpatialAnalyzer(llm)
        self.reasoner = FieldWorkReasoner(config, llm)
        self.formatter = AnswerFormatter()

    async def handle(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
        updater: TaskUpdater,
    ) -> str:
        """
        Process a FieldWorkArena task end-to-end.

        Args:
            text: Goal text from green agent (# Question / # Input Data / # Output Format)
            file_parts: List of (name, mime_type, data) for attached files
            updater: A2A task updater for progress reporting

        Returns:
            Formatted answer string
        """
        # 1. Parse the goal string
        task = self.parser.parse(text)
        logger.info(f"Task parsed: query='{task.query[:80]}...', files={len(file_parts)}")

        # 2. Process all file attachments into text context
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Processing {len(file_parts)} input file(s)..."
            ),
        )

        file_contexts = []
        for name, mime, data in file_parts:
            context = await self.vision.process_file(name, mime, data)
            file_contexts.append(context)

        logger.info(f"Processed {len(file_contexts)} files into context")

        # 3. Build spatial scene graph
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Building spatial scene graph..."),
        )

        scene = await self.spatial.build_scene(task.query, file_contexts)
        logger.info(
            f"Scene: {scene.entity_count} entities, "
            f"{len(scene.relations)} relations, "
            f"{scene.violation_count} violations"
        )

        # 4. Reason over evidence to produce answer
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reasoning over evidence..."),
        )

        answer = await self.reasoner.reason(
            query=task.query,
            file_contexts=file_contexts,
            scene=scene,
            output_format=task.output_format,
        )

        # 5. Format answer to match expected output
        formatted = self.formatter.format_answer(answer, task.output_format)
        logger.info(f"Answer formatted ({len(formatted)} chars): {formatted[:200]}...")

        return formatted
