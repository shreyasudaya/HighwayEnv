import gymnasium  
import highway_env
import matplotlib.pyplot as plt
env = gymnasium.make(
  "zippermerge-v0",
  render_mode="rgb_array",
  config={
    "observation": {
      "type": "MultiAgentObservation",
      "observation_config": {
        "type": "Kinematics",
      }
    }
  }
)
env.unwrapped.config.update({
  "action": {
    "type": "MultiAgentAction",
    "action_config": {
      "type": "DiscreteMetaAction",
    }
  }
})
env.reset()

_, (ax1, ax2) = plt.subplots(nrows=2)
ax1.imshow(env.render())
ax1.set_title("Initial state")

# Make the first vehicle change to the left lane, and the second one to the right
action_1, action_2 = 0, 2  # See highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL
env.step((action_1, action_2))

ax2.imshow(env.render())
ax2.set_title("After sending actions to each vehicle")
plt.show()