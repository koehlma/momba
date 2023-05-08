- `dtmc/brp`: Wrong number of transitions?
- `dtmc/herman`: Unspecified initial values?



# Requires Functions
- `dtmc/egl`
- `mdp/pacman`
- `mdp/resource-gathering`
- `mdp/wlan`
- `mdp/wlan_dl`

# No Translation
- `dtmc/oscillators`


# Fixed

## Error in Bit Encoding (`lower_bound != 0`)
- `dtmc/nand`: Wrong number of states (too few)!
- `mdp/echoring`: Wrong number of states (too few)!
- `mdp/rabin`: Wrong number of states (too few)!
- `dtmc/leader_sync`: Wrong number of states (too many)!
- `mdp/zeroconf`: Wrong number of states (with `reset=False`)!
- `mdp/zeroconf_dl`: Wrong number of states (with `reset=False`)!