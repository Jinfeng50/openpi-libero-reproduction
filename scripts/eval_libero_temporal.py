#!/usr/bin/env python3
"""Evaluate an openpi policy on LIBERO with baseline or DGTE action execution.

Run this from the openpi environment because the LIBERO simulator and
``openpi_client`` are provided there.  The policy server remains unchanged;
only the client-side action-chunk execution strategy is varied.
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import pathlib
import sys

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tqdm

PERSONAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERSONAL_ROOT / "src"))
from openpi_libero_reproduction.temporal_ensemble import (  # noqa: E402
    DGTEConfig,
    DisagreementGatedTemporalEnsembler,
)
from openpi_libero_reproduction.transition_dataset import EpisodeTransitionRecorder  # noqa: E402
from openpi_libero_reproduction.world_model_controller import WorldModelActionSelector  # noqa: E402
from openpi_libero_reproduction.world_model_controller import align_action_chunk  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--video-out-path", default="data/libero/videos")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--controller", choices=("baseline", "dgte", "world_model"), default="dgte")
    parser.add_argument("--dgte-decay", type=float, default=0.7)
    parser.add_argument("--dgte-disagreement-threshold", type=float, default=0.08)
    parser.add_argument("--dgte-gate-strength", type=float, default=3.0)
    parser.add_argument(
        "--record-transitions",
        type=pathlib.Path,
        default=None,
        help="write one episode-level NPZ shard per rollout for world-model training",
    )
    parser.add_argument("--world-model-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--world-model-device", default="cpu")
    parser.add_argument("--world-model-encoder-weights", choices=("default", "none"), default="default")
    parser.add_argument("--world-model-uncertainty-penalty", type=float, default=0.1)
    return parser.parse_args()


def eval_libero(args: argparse.Namespace) -> tuple[float, int]:
    np.random.seed(args.seed)
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be positive")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    dgte_config = DGTEConfig(
        replan_steps=args.replan_steps,
        decay=args.dgte_decay,
        disagreement_threshold=args.dgte_disagreement_threshold,
        gate_strength=args.dgte_gate_strength,
    )
    world_model_selector = None
    if args.controller == "world_model":
        if args.world_model_checkpoint is None:
            raise ValueError("--world-model-checkpoint is required with --controller world_model")
        world_model_selector = WorldModelActionSelector(
            args.world_model_checkpoint,
            device=args.world_model_device,
            encoder_weights=args.world_model_encoder_weights,
            uncertainty_penalty=args.world_model_uncertainty_penalty,
        )

    total_episodes = 0
    total_successes = 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc=args.task_suite_name):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_episodes = 0
        task_successes = 0
        try:
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task {task_id}", leave=False):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                action_plan: collections.deque[np.ndarray] = collections.deque()
                controller = (
                    DisagreementGatedTemporalEnsembler(dgte_config)
                    if args.controller in {"dgte", "world_model"}
                    else None
                )
                if controller is not None:
                    controller.reset()
                recorder = (
                    EpisodeTransitionRecorder(
                        args.record_transitions,
                        suite=args.task_suite_name,
                        controller=args.controller,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        prompt=str(task_description),
                        seed=args.seed,
                        replan_steps=args.replan_steps,
                    )
                    if args.record_transitions is not None
                    else None
                )
                pending_transition = None
                replay_images: list[np.ndarray] = []
                done = False
                t = 0
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img, wrist_img, state = _prepare_observation(obs, args.resize_size)
                    replay_images.append(img)

                    if not action_plan:
                        if pending_transition is not None:
                            _record_transition(
                                recorder,
                                pending_transition,
                                future_image=img,
                                future_wrist_image=wrist_img,
                                future_state=state,
                                future_step=t,
                                terminal_within_horizon=False,
                            )
                            pending_transition = None
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }
                        action_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)
                        if action_chunk.ndim != 2 or len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"policy returned {action_chunk.shape}; expected at least "
                                f"{args.replan_steps} actions"
                            )
                        if args.controller == "baseline":
                            selected_actions = action_chunk[: args.replan_steps]
                        elif args.controller == "world_model":
                            assert controller is not None and world_model_selector is not None
                            controller.add_chunk(action_chunk, start_step=t)
                            candidates = controller.chunks_covering(t)
                            scoring_chunks = [
                                align_action_chunk(chunk, t - source, world_model_selector.action_horizon)
                                for source, chunk in candidates
                            ]
                            _, scores = world_model_selector.select_chunk(
                                image=img,
                                wrist_image=wrist_img,
                                state=state,
                                prompt=str(task_description),
                                action_chunks=scoring_chunks,
                            )
                            selected_index = int(np.argmax(scores))
                            selected_source, selected_chunk = candidates[selected_index]
                            selected_offset = t - selected_source
                            selected_actions = selected_chunk[
                                selected_offset : selected_offset + args.replan_steps
                            ]
                            if len(selected_actions) != args.replan_steps:
                                raise RuntimeError(
                                    "selected world-model chunk does not cover the full replan horizon"
                                )
                            controller.prune_before(t + args.replan_steps)
                        else:
                            assert controller is not None
                            controller.add_chunk(action_chunk, start_step=t)
                            selected_actions = controller.next_actions(t, args.replan_steps)
                        pending_transition = {
                            "image": img.copy(),
                            "wrist_image": wrist_img.copy(),
                            "state": state.copy(),
                            "action_chunk": action_chunk.copy(),
                            "selected_actions": np.asarray(selected_actions, dtype=np.float32).copy(),
                            "executed_steps": 0,
                            "start_step": t,
                        }
                        action_plan.extend(selected_actions)

                    obs, _, done, _ = env.step(action_plan.popleft().tolist())
                    if pending_transition is not None:
                        pending_transition["executed_steps"] += 1
                    if done:
                        if pending_transition is not None:
                            future_img, future_wrist_img, future_state = _prepare_observation(obs, args.resize_size)
                            _record_transition(
                                recorder,
                                pending_transition,
                                future_image=future_img,
                                future_wrist_image=future_wrist_img,
                                future_state=future_state,
                                future_step=t + 1,
                                terminal_within_horizon=True,
                            )
                            pending_transition = None
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                if pending_transition is not None:
                    future_img, future_wrist_img, future_state = _prepare_observation(obs, args.resize_size)
                    _record_transition(
                        recorder,
                        pending_transition,
                        future_image=future_img,
                        future_wrist_image=future_wrist_img,
                        future_state=future_state,
                        future_step=pending_transition["start_step"] + pending_transition["executed_steps"],
                        terminal_within_horizon=False,
                    )
                if recorder is not None:
                    shard = recorder.finish(episode_success=done)
                    if shard is not None:
                        logging.info("Recorded transition shard: %s", shard)

                task_episodes += 1
                total_episodes += 1
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"task_{task_id:02d}_episode_{episode_idx:03d}_{task_segment}_{suffix}.mp4",
                    replay_images,
                    fps=10,
                )
                logging.info("Success: %s", done)
        finally:
            env.close()
        logging.info(
            "Task %s success rate: %.4f (%d/%d)",
            task_id,
            task_successes / task_episodes if task_episodes else 0.0,
            task_successes,
            task_episodes,
        )

    success_rate = total_successes / total_episodes if total_episodes else 0.0
    logging.info("Controller: %s", args.controller)
    logging.info("Total success rate: %.4f", success_rate)
    logging.info("Total episodes: %d", total_episodes)
    return success_rate, total_episodes


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _prepare_observation(obs: dict, resize_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert simulator images/state to the exact policy input representation."""

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32, copy=False)
    return img, wrist_img, state


def _record_transition(
    recorder: EpisodeTransitionRecorder | None,
    pending: dict,
    *,
    future_image: np.ndarray,
    future_wrist_image: np.ndarray,
    future_state: np.ndarray,
    future_step: int,
    terminal_within_horizon: bool,
) -> None:
    if recorder is None:
        return
    recorder.add(
        image=pending["image"],
        wrist_image=pending["wrist_image"],
        future_image=future_image,
        future_wrist_image=future_wrist_image,
        state=pending["state"],
        future_state=future_state,
        action_chunk=pending["action_chunk"],
        selected_actions=pending["selected_actions"],
        executed_steps=pending["executed_steps"],
        start_step=pending["start_step"],
        future_step=future_step,
        terminal_within_horizon=terminal_within_horizon,
    )


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3])))) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(parse_args())
