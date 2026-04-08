import os
import torch
import numpy as np
from environment import EmailEnv
from agent import DQNAgent

def train():
    # Initialize environment and agent
    env = EmailEnv()
    state_size = 5 # urgency, sentiment, spam_score, wait_time, confidence
    action_size = 4 # open, delete, defer, escalate
    agent = DQNAgent(state_size, action_size)
    
    batch_size = 32
    episodes = 100
    model_dir = "src/models"
    model_path = os.path.join(model_dir, "dqn_email_v1.pth")
    
    print(f"Starting training for {episodes} episodes...")
    
    for e in range(episodes):
        state = env.reset()
        # Convert state dict to vector
        state_v = np.array([
            state.get("urgency_val", 0.5),
            state.get("sentiment_val", 0.5),
            state.get("spam_score", 0.0),
            state.get("wait_norm", 0.0),
            state.get("confidence", 0.5)
        ])
        
        total_reward = 0
        for time in range(50): # 50 emails per episode
            action = agent.get_action(state_v)
            
            # Map action index to type string
            action_map = ["open", "delete", "defer", "escalate"]
            action_type = action_map[action]
            
            next_state, reward, done, _ = env.step(action_type)
            
            next_state_v = np.array([
                next_state.get("urgency_val", 0.5),
                next_state.get("sentiment_val", 0.5),
                next_state.get("spam_score", 0.0),
                next_state.get("wait_norm", 0.0),
                next_state.get("confidence", 0.5)
            ])
            
            agent.remember(state_v, action, reward, next_state_v, done)
            state_v = next_state_v
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                break
                
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Save model every 10 episodes
        if (e + 1) % 10 == 0:
            agent.save(model_path)
            print(f"Saved model to {model_path}")

    print("Training finished!")

if __name__ == "__main__":
    train()
