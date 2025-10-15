# training_manual.py
import csv
import os
import pygame
import numpy as np
from rocket_env import SimpleRocketEnv

def main(output_csv="manual_data.csv", n_episodes=5):
    env = SimpleRocketEnv(render_mode="human")

    # CSV header
    fieldnames = [
        "episode", "step", "x", "y", "vx", "vy",
        "sin_theta", "cos_theta", "omega", "dx", "dy",
        "action", "reward", "done"
    ]

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("\n=== Kontrol Manual ===")
        print("SPACE = main thrust | LEFT = left thruster | RIGHT = right thruster")
        print("ESC = keluar episode\n")

        for ep in range(1, n_episodes + 1):
            state, _ = env.reset()
            done = False
            step = 0
            prev_reward = 0.0  # ← reward step sebelumnya

            print(f"--- Episode {ep} ---")

            while not done:
                action = 0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
                        elif event.key == pygame.K_SPACE:
                            action = 1
                        elif event.key == pygame.K_RIGHT:
                            action = 2
                        elif event.key == pygame.K_LEFT:
                            action = 3

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                step += 1

                # save next_state (real values)
                norm_next = next_state * env._normalizer()
                row = {
                    "episode": ep, "step": step,
                    "x": norm_next[0], "y": norm_next[1],
                    "vx": norm_next[2], "vy": norm_next[3],
                    "sin_theta": norm_next[4], "cos_theta": norm_next[5],
                    "omega": norm_next[6], "dx": norm_next[7], "dy": norm_next[8],
                    "action": action,
                    "reward": reward,
                    "done": done
                }

                # jika ini step terakhir, ganti reward dengan prev_reward
                if done:
                    row["reward"] = prev_reward

                writer.writerow(row)
                print(f"Step {step} | action {action} | reward {reward:.3f}")

                prev_reward = reward  # update reward sebelumnya
                env.render()
                state = next_state

            print(f"Episode {ep} selesai.\n")

    env.close()
    print(f"\n✅ Semua {n_episodes} episode disimpan di '{output_csv}'")

if __name__ == "__main__":
    main("manual_data.csv", n_episodes=5)
