import base64
import io
import json
import logging
import textwrap
from typing import Any, Dict, List, Literal

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from ..structs import FrameData, GameAction
from .llm_agents import ReasoningLLM

logger = logging.getLogger(__name__)

class ReasoningActionResponse(BaseModel):
    name: Literal["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"] = Field(
        description="The action to take."
    )
    aggregated_findings: str = Field(
        description="Summary of discoveries and learnings so far.",
        min_length=10,
        max_length=2000,
    )
    victory_hypothesis: str = Field(
        description="Current best theory on how to win the level/game.",
        min_length=10,
        max_length=1000,
    )
    hypothesis: str = Field(
        description="Current hypothesis to test about game mechanics.",
        min_length=10,
        max_length=2000,
    )
    plan: str = Field(
        description="Plan in natural language: explain the actions you will do to test your hypothesis.",
        min_length=10,
        max_length=1000,
    )
    planned_actions: List[str] = Field(
        description="Ordered list of action names to take for this experiment, e.g. ['ACTION1', 'ACTION1', 'ACTION4']"
    )
    score_progress: str = Field(
        description="Describe whether your score increased after the last sequence, and what you believe caused it.",
        min_length=5,
        max_length=300,
        default="No score progress detected."
    )

class ReasoningAgent(ReasoningLLM):
    MAX_ACTIONS = 60
    DO_OBSERVATION = True
    MODEL = "o4-mini"
    MESSAGE_LIMIT = 5
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: List[ReasoningActionResponse] = []
        self.screen_history: List[bytes] = []
        self.max_screen_history = 10
        self.client = OpenAI()
        self.action_queue: List[str] = []
        self.current_reasoning: ReasoningActionResponse = None
        self.previous_score: int = 0

    def clear_history(self) -> None:
        self.history = []
        self.screen_history = []
        self.action_queue = []
        self.current_reasoning = None
        self.previous_score = 0

    def generate_grid_image_with_zone(
        self, grid: List[List[int]], cell_size: int = 40, zone_size: int = 20
    ) -> bytes:
        if not grid or not grid[0]:
            img = Image.new("RGB", (200, 200), color="black")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        height = len(grid)
        width = len(grid[0])
        img = Image.new("RGB", (width * cell_size, height * cell_size), color="white")
        draw = ImageDraw.Draw(img)
        key_colors = {
            0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999", 3: "#666666", 4: "#333333", 5: "#000000",
            6: "#E53AA3", 7: "#FF7BCC", 8: "#F93C31", 9: "#1E93FF", 10: "#88D8F1", 11: "#FFDC00",
            12: "#FF851B", 13: "#921231", 14: "#4FCC30", 15: "#A356D6"
        }
        for y in range(height):
            for x in range(width):
                color = key_colors.get(grid[y][x], "#888888")
                draw.rectangle(
                    [
                        x * cell_size,
                        y * cell_size,
                        (x + 1) * cell_size,
                        (y + 1) * cell_size,
                    ],
                    fill=color,
                    outline="#000000",
                    width=1,
                )
        for y in range(0, height, zone_size):
            for x in range(0, width, zone_size):
                try:
                    font = ImageFont.load_default()
                    zone_text = f"({x},{y})"
                    draw.text(
                        (x * cell_size + 2, y * cell_size + 2),
                        zone_text,
                        fill="#FFFFFF",
                        font=font,
                    )
                except (ImportError, OSError) as e:
                    logger.debug(f"Could not load font for zone labels: {e}")
                except Exception as e:
                    logger.error(f"Failed to draw zone label at ({x},{y}): {e}")
                zone_width = min(zone_size, width - x) * cell_size
                zone_height = min(zone_size, height - y) * cell_size
                draw.rectangle(
                    [
                        x * cell_size,
                        y * cell_size,
                        x * cell_size + zone_width,
                        y * cell_size + zone_height,
                    ],
                    fill=None,
                    outline="#FFD700",
                    width=2,
                )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def build_functions(self) -> list[dict[str, Any]]:
        schema = ReasoningActionResponse.model_json_schema()
        schema["properties"].pop("name", None)
        if "required" in schema:
            schema["required"].remove("name")
        functions: list[dict[str, Any]] = [
            {
                "name": action.name,
                "description": f"Take action {action.name}",
                "parameters": schema,
            }
            for action in [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.ACTION5,
                GameAction.ACTION6,
                GameAction.RESET,
            ]
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                    },
                }
            )
        return tools

    def build_user_prompt(self, latest_frame: FrameData, score_progress: str = "No score progress detected.") -> str:
        return textwrap.dedent(
            f"""
You are an agent playing a dynamic video game. You are curious and eager to understand how the game works by interacting with its different objects and elements. Your goal is to WIN and avoid GAME_OVER while minimizing actions.

You can do 5 actions:
- RESET (used to start a new game or level)
- ACTION1 (MOVE_UP)
- ACTION2 (MOVE_DOWN)
- ACTION3 (MOVE_LEFT)
- ACTION4 (MOVE_RIGHT)

After each sequence, always output your reasoning fields in this order:
0. aggregated_findings: what you have discovered so far about game logic and objects.
0. victory_hypothesis: your best current theory about how to win the current level/game.
1. hypothesis: the hypothesis you are about to test. Always make your hypothesis about a **specific object or element** you see in the game (for example: key, door, switch, enemy...).
2. plan: explain, in natural language, your plan to test this hypothesis (sequence of moves), and clearly state **which object or element you are interacting with and why**.
3. planned_actions: explicit list of action codes (e.g. ["ACTION3", "ACTION3", "ACTION2", "ACTION5"]).
4. score_progress: say if your score increased after the last plan, and what caused it.

IMPORTANT:
- Treat this as an interactive game: be **curious** and try to interact with visible elements to discover their function.
- After every planned sequence, check if your score increased compared to before the plan.
- If score increased, explain why you believe it happened.
- Always update aggregated_findings, hypothesis, and victory_hypothesis after new evidence.
- **Never propose a plan or hypothesis that is not related to a specific object or element in the game.**
- Avoid proposing sequences of actions that just move around randomly or repeat the same action without a clear target or goal.
- If you have already tested all visible elements, try combinations or look for new interactions.

Current score info: {score_progress}

Example:
aggregated_findings: "- Keys are collected by stepping on purple tiles. - Doors can only be opened after collecting all keys."
victory_hypothesis: "Win = collect all keys then open the door with ACTION5."
hypothesis: "Collecting the purple key will update my inventory and allow the door to be opened."
plan: "Move three times left, then down, to reach the key; then use interact to try to open the door."
planned_actions: ["ACTION3", "ACTION3", "ACTION3", "ACTION2", "ACTION5"]
score_progress: "Score increased from 1 to 2 after the last sequence (due to collecting the key)."

Your task:
- Experiment with the game by performing and analyzing sequences of actions, based on your hypothesis and plan.
- Be curious! Your hypotheses and plans should always involve interacting with objects or elements you can see on the screen.
- After each sequence, analyze the impact by comparing screens before/after, and by checking score.
- Summarize and update your findings so your colleagues can learn the rules.

Remember: Always PLAN a sequence (not a single action), describe it clearly, and list the corresponding planned_actions array.
"""
        )

    def call_llm_with_structured_output(
        self, messages: List[Dict[str, Any]]
    ) -> ReasoningActionResponse:
        try:
            tools = self.build_tools()
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
            self.track_tokens(
                response.usage.total_tokens, response.choices[0].message.content
            )
            self.capture_reasoning_from_response(response)

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            if tool_calls:
                tool_call = tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                function_args["name"] = tool_call.function.name
                return ReasoningActionResponse(**function_args)

            raise ValueError("LLM did not return a tool call.")

        except Exception as e:
            logger.error(f"LLM structured call failed: {e}")
            raise e

    def define_next_action(self, latest_frame: FrameData, score_progress: str = "No score progress detected.") -> ReasoningActionResponse:
        current_grid = latest_frame.frame[-1] if latest_frame.frame else []
        map_image = self.generate_grid_image_with_zone(current_grid)
        system_prompt = self.build_user_prompt(latest_frame, score_progress)
        latest_action = self.history[-1] if self.history else None
        user_message_content: List[Dict[str, Any]] = []
        previous_screen = self.screen_history[-1] if self.screen_history else None

        if previous_screen:
            user_message_content.extend(
                [
                    {"type": "text", "text": "Previous screen:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(previous_screen).decode()}",
                            "detail": "high",
                        },
                    },
                ]
            )

        raw_grid_text = self.pretty_print_3d(latest_frame.frame)
        user_message_text = f"Your previous action was: {json.dumps(latest_action.model_dump() if latest_action else None, indent=2)}\n\nAttached are the visual screen and raw grid data.\n\nRaw Grid:\n{raw_grid_text}\n\nWhat should you do next?"

        current_image_b64 = base64.b64encode(map_image).decode()
        user_message_content.extend(
            [
                {"type": "text", "text": user_message_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{current_image_b64}",
                        "detail": "high",
                    },
                },
            ]
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content},
        ]

        result = self.call_llm_with_structured_output(messages)

        self.screen_history.append(map_image)
        if len(self.screen_history) > self.max_screen_history:
            self.screen_history.pop(0)

        return result

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Détecter progression de score
        score_progress = "No score progress detected."
        if hasattr(self, "previous_score"):
            if latest_frame.score > self.previous_score:
                score_progress = f"Score increased from {self.previous_score} to {latest_frame.score} after your last sequence."
            elif latest_frame.score < self.previous_score:
                score_progress = f"Score decreased from {self.previous_score} to {latest_frame.score} (unexpected, investigate!)."
        else:
            self.previous_score = latest_frame.score

        if latest_frame.full_reset:
            self.clear_history()
            return GameAction.RESET

        if not self.history:
            action = GameAction.RESET
            initial_response = ReasoningActionResponse(
                name="RESET",
                aggregated_findings="No findings yet.",
                victory_hypothesis="The game requires a RESET to begin.",
                hypothesis="The game requires a RESET to begin.",
                plan="Reset the game to begin exploring the mechanics.",
                planned_actions=["RESET"],
                score_progress="No score progress detected."
            )
            self.history.append(initial_response)
            self.action_queue = []
            self.current_reasoning = initial_response
            self.previous_score = latest_frame.score
            return action

        # Si un plan est en cours, on continue
        if self.action_queue:
            action_name = self.action_queue.pop(0)
            action = GameAction.from_name(action_name)
            action.reasoning = {
                "plan_in_progress": True,
                "remaining": self.action_queue.copy(),
            }
            self.previous_score = latest_frame.score
            return action

        # Nouveau plan : score_progress transmis au LLM pour explication !
        action_response = self.define_next_action(latest_frame, score_progress)
        self.history.append(action_response)
        self.action_queue = list(action_response.planned_actions)
        self.current_reasoning = action_response
        if not self.action_queue:
            action = GameAction.RESET
            action.reasoning = {"error": "No planned actions from LLM, forced RESET"}
            self.current_reasoning = None
            self.previous_score = latest_frame.score
            return action

        # Premier step du nouveau plan : reasoning complet, score_progress
        action_name = self.action_queue.pop(0)
        action = GameAction.from_name(action_name)
        action.reasoning = {
            "plan": action_response.planned_actions,
            "aggregated_findings": action_response.aggregated_findings,
            "victory_hypothesis": action_response.victory_hypothesis,
            "hypothesis": action_response.hypothesis,
            "plan_natural": action_response.plan,
            "score_progress": score_progress,
            "plan_in_progress": len(self.action_queue) > 0
        }
        self.previous_score = latest_frame.score
        return action
