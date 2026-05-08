# 6 nimmt! Open Tournament Server
# Copyright (C) 2026  Jhong-Ken Chen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import asyncio
import json
import logging
import ssl
import sys
import importlib
import signal
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TournamentClient")


class TournamentClient:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.server_host = self.config.get("server_host", "127.0.0.1")
        self.server_port = self.config.get("server_port", 8888)
        self.username = self.config.get("username", "test_user")
        self.password = self.config.get("password", "test_pass")
        self.label = self.config.get("label", "agent-v1")

        # Load player class
        player_config = self.config.get("player", {})
        self.player_path = player_config.get("path")
        self.player_class_name = player_config.get("class")
        self.player_args = player_config.get("args", {})

        self.PlayerClass = self._load_player_class(
            self.player_path, self.player_class_name
        )

        self.reader = None
        self.writer = None
        self.player_instance = None
        self.player_index = None
        self.running = True
        self.in_game = False
        self.pending_exit = False
        self.ctrl_c_count = 0

    def _load_player_class(self, path, class_name):
        try:
            module = importlib.import_module(path)
            cls = getattr(module, class_name)
            logger.info(f"Successfully loaded {class_name} from {path}")
            return cls
        except Exception as e:
            logger.error(f"Failed to load player class: {e}")
            sys.exit(1)

    async def connect(self):
        while self.running:
            try:
                logger.info(f"Connecting to {self.server_host}:{self.server_port}...")

                # For TLS
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = (
                    ssl.CERT_NONE
                )  # Accept self-signed certs for the tournament

                self.reader, self.writer = await asyncio.open_connection(
                    self.server_host, self.server_port, ssl=ssl_context
                )
                logger.info("Connected successfully.")

                # Setup signals
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.add_signal_handler(sig, self._handle_sigint)
                    except NotImplementedError:
                        pass  # Windows

                await self._login()
                await self._message_loop()

            except (ConnectionRefusedError, ConnectionResetError, ssl.SSLError) as e:
                logger.warning(f"Connection error: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            except asyncio.IncompleteReadError:
                logger.warning("Disconnected from server. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(5)
            finally:
                if self.writer:
                    self.writer.close()
                    try:
                        await self.writer.wait_closed()
                    except:
                        pass

    def _handle_sigint(self):
        self.ctrl_c_count += 1
        if self.ctrl_c_count >= 2:
            print("\nForce exiting...")
            os._exit(1)

        if not self.in_game:
            print("\nCtrl-C detected. Exiting...")
            self.running = False
            if self.writer:
                self.writer.close()
        else:
            print("\nCtrl-C detected. Will exit gracefully after this match.")
            self.pending_exit = True

    async def _send_msg(self, msg_dict):
        msg_str = json.dumps(msg_dict) + "\n"
        self.writer.write(msg_str.encode("utf-8"))
        await self.writer.drain()
        logger.debug(f"Sent: {msg_dict['type']}")

    async def _login(self):
        login_msg = {
            "type": "login",
            "user": self.username,
            "pass": self.password,
            "label": self.label,
        }
        await self._send_msg(login_msg)

    async def _message_loop(self):
        while True:
            line = await self.reader.readuntil(b"\n")
            if not line:
                break

            try:
                msg = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                logger.error(f"Received invalid JSON: {line}")
                continue

            msg_type = msg.get("type")
            logger.debug(f"Received: {msg_type}")

            if msg_type == "login_success":
                logger.info("Login successful. Sending ready signal...")
                await self._send_msg({"type": "ready"})
            elif msg_type == "login_failed":
                logger.error(f"Login failed: {msg.get('reason')}")
                self.running = False
                break
            elif msg_type == "match_start":
                await self._handle_match_start(msg)
            elif msg_type == "request_action":
                await self._handle_request_action(msg)
            elif msg_type == "match_over":
                self.in_game = False
                if self.pending_exit:
                    logger.info("Exiting gracefully as requested.")
                    self.running = False
                    break
                logger.info("Match over. Sending ready signal for next match...")
                self.player_instance = None
                self.game_id = None
                await self._send_msg({"type": "ready"})
            else:
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_match_start(self, msg):
        self.game_id = msg.get("game_id")
        self.player_index = msg.get("player_index")
        self.in_game = True
        logger.info(
            f"Match started! Label: {self.label}, Game ID: {self.game_id}, Player Index: {self.player_index}"
        )

        try:
            # Initialize player instance
            self.player_instance = self.PlayerClass(
                player_idx=self.player_index, **self.player_args
            )
            await self._send_msg({"type": "init_ready"})
            logger.info("Player initialized and ready.")
        except Exception as e:
            logger.error(f"Error initializing player: {e}")
            # Even if it fails, we shouldn't crash the client wrapper.
            # We just let it timeout or fail on the server side, or we can send an error message if the protocol supports it.

    async def _handle_request_action(self, msg):
        hand = msg.get("hand", [])
        history_state = msg.get("history_state", {})
        round_idx = history_state.get("round", 0)

        logger.info(f"Round {round_idx} started. Hand: {hand}")

        if not self.player_instance:
            logger.error("Player instance not found, but action requested.")
            # Fallback to smallest card just to send something
            fallback_card = min(hand) if hand else 0
            await self._send_msg({"type": "action", "card": fallback_card})
            return

        try:
            # Call action (this is synchronous, so we use asyncio.to_thread to avoid blocking the event loop)
            # The TA engine timeout is 1.2s, so we shouldn't take longer than that
            card = await asyncio.to_thread(
                self.player_instance.action, hand, history_state
            )

            logger.info(f"Action determined: {card}")
            await self._send_msg({"type": "action", "card": int(card)})
        except Exception as e:
            logger.error(f"Error during action: {e}")
            # If the player crashes, fallback
            fallback_card = min(hand) if hand else 0
            await self._send_msg({"type": "action", "card": fallback_card})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.client <config_path>")
        sys.exit(1)

    config_file = sys.argv[1]
    if not Path(config_file).exists():
        print(f"Config file not found: {config_file}")
        sys.exit(1)

    client = TournamentClient(config_file)

    try:
        asyncio.run(client.connect())
    except KeyboardInterrupt:
        print("\nClient terminated by user.")
