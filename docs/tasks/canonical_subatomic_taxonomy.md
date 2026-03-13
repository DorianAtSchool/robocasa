## Canonical Subatomic Taxonomy

This document defines a more decomposable subatomic layer for RoboCasa atomic tasks.
It normalizes the current 65-task atomic catalog into a smaller set of reusable
subatomic tasks for planning, tool naming, and future dataset refactoring.

### Scope

- This is a planning taxonomy, not a requirement that every subatomic task already exist
  as a separate RoboCasa environment.
- The canonical layer should stay parameterized.
- Fixture-specific names such as `open_fridge_door` or `place_in_cabinet` are useful
  task-facing aliases, but they should compile down to a smaller canonical subatomic set.

### Design Rules

- One subatomic task should correspond to one externally verifiable state change.
- Source, destination, manipulated part, and goal state should be explicit arguments.
- Multi-part fixtures should keep the manipulated `part` explicit, for example
  `door_left`, `door_right`, `rack_top`, or `rack_bottom`.
- Transport should be decomposed into `pick_up_object(...)` plus a placement subatomic task,
  instead of hiding both steps inside a fixture-pair task name.
- Access should be explicit. If a current RoboCasa pick-place task silently initializes
  a cabinet, drawer, or appliance as open, that hidden setup should become an explicit
  `open_*` subatomic task at task-planning time.
- Directional tasks should be split explicitly. A subatomic task should never hide a choice
  like `{left,right}`, `{in,out}`, or `{increase,decrease}` in the initial state.

### Canonical Subatomic Tasks

`support` means an exposed surface or attachment seat. `receptacle` means an interior
or container-like region that contains the object.

| Family | Subatomic task | Signature | Semantics | Example aliases |
| --- | --- | --- | --- | --- |
| Navigation | `navigate_to_fixture` | `navigate_to_fixture(fixture)` | Move the robot base to the working pose of a fixture. | `navigate_to_fridge`, `navigate_to_sink` |
| Transport | `pick_up_object` | `pick_up_object(object, source)` | Grasp and lift an object from its current source region. | `pickup_from_counter`, `pickup_from_cabinet` |
| Transport | `place_on_surface` | `place_on_surface(object, support)` | Place an object onto an exposed support surface or attachment seat. | `place_on_counter`, `place_on_stove_burner` |
| Transport | `place_in_receptacle` | `place_in_receptacle(object, receptacle)` | Place an object into an interior or container-like region. | `place_in_cabinet`, `place_in_microwave`, `place_in_tupperware` |
| Transport | `place_on_object` | `place_on_object(object, support_object)` | Place an object on top of another movable object. | `place_cheese_on_bread` |
| Transport | `place_under_dispenser` | `place_under_dispenser(object, dispenser)` | Place an object under a machine dispenser. | `place_mug_under_coffee_dispenser` |
| Access | `open_hinged_part` | `open_hinged_part(target, part)` | Open a hinged door, lid, or head. | `open_fridge_door`, `open_stand_mixer_head` |
| Access | `close_hinged_part` | `close_hinged_part(target, part)` | Close a hinged door, lid, or head. | `close_cabinet_door`, `close_kettle_lid` |
| Access | `open_sliding_part` | `open_sliding_part(target, part)` | Pull out a sliding drawer or rack. | `open_drawer`, `pull_out_oven_rack` |
| Access | `close_sliding_part` | `close_sliding_part(target, part)` | Push in a sliding drawer or rack. | `close_drawer`, `push_in_dishwasher_rack` |
| Controls | `press_button` | `press_button(target, control)` | Activate a discrete button-like control. | `press_coffee_machine_button`, `press_microwave_start` |
| Controls | `press_lever` | `press_lever(target, control)` | Activate a discrete lever-like control. | `press_toaster_lever`, `press_kettle_power_lever` |
| Controls | `set_rotary_control` | `set_rotary_control(target, control, goal)` | Set a knob, handle, or rotatable control to an explicit goal state. | `turn_stove_burner_on`, `turn_sink_spout_left`, `increase_toaster_oven_temperature` |

### Recommended Alias Layer

The canonical layer should stay small. Human-readable task-facing names can be more
specific as long as they lower into the canonical subatomic tasks above.

Recommended alias patterns:

- Access:
  - `open_cabinet_door`
  - `close_fridge_door`
  - `open_fridge_drawer`
  - `pull_out_dishwasher_rack`
  - `push_in_toaster_oven_rack`
- Transport:
  - `pickup_from_counter`
  - `pickup_from_cabinet`
  - `place_on_counter`
  - `place_in_cabinet`
  - `place_in_microwave`
  - `place_under_coffee_dispenser`
  - `place_on_bread`
- Controls:
  - `press_coffee_machine_button`
  - `press_microwave_stop`
  - `press_toaster_lever`
  - `turn_stove_burner_low`
  - `turn_sink_faucet_on`
  - `turn_sink_spout_right`
  - `increase_toaster_oven_temperature`

### Coverage Summary

| Current RoboCasa atomic family | Count | Canonical treatment |
| --- | --- | --- |
| Navigation | 1 | Mostly 1:1 |
| Transport and placement | 23 | Split into `pick_up_object(...)` + placement |
| Access and articulation | 25 | Mostly 1:1, with explicit direction where needed |
| Controls and activation | 16 | Mostly 1:1, with explicit goal states |
| Total current atomic tasks | 65 | Fully covered below |

### Mapping Notes

- The mapping below captures the behavioral core of each current RoboCasa atomic task.
- Several current pick-place tasks auto-open cabinets, drawers, or appliances in the
  environment setup. That hidden access step is not preserved as part of the transport
  subatomic sequence; task-level planners should add the corresponding access subatomic task
  explicitly when the world state requires it.
- `Slide*`, `TurnSinkSpout`, and `AdjustToasterOvenTemperature` should be split into
  direction-explicit aliases even though the current RoboCasa task classes hide that
  direction in episode state.

### Navigation Mapping

| Current RoboCasa atomic task | Canonical subatomic sequence | Suggested explicit alias |
| --- | --- | --- |
| `NavigateKitchen` | `navigate_to_fixture(target_fixture)` | `navigate_to_fixture` |

### Transport and Placement Mapping

All tasks in this section should stop being leaf atomic tasks in a canonical planner.
They should decompose into a pickup subatomic task plus a placement subatomic task.

| Current RoboCasa atomic task | Canonical subatomic sequence | Suggested explicit alias |
| --- | --- | --- |
| `CoffeeServeMug` | `pick_up_object(mug, coffee_machine_dispenser)` + `place_on_surface(mug, counter)` | `pickup_from_coffee_dispenser` + `place_on_counter` |
| `CoffeeSetupMug` | `pick_up_object(mug, counter)` + `place_under_dispenser(mug, coffee_machine_dispenser)` | `pickup_from_counter` + `place_under_coffee_dispenser` |
| `PickPlaceCabinetToCounter` | `pick_up_object(object, cabinet_interior)` + `place_on_surface(object, counter)` | `pickup_from_cabinet` + `place_on_counter` |
| `PickPlaceCounterToBlender` | `pick_up_object(object, counter)` + `place_in_receptacle(object, blender_jar)` | `pickup_from_counter` + `place_in_blender` |
| `PickPlaceCounterToCabinet` | `pick_up_object(object, counter)` + `place_in_receptacle(object, cabinet_interior)` | `pickup_from_counter` + `place_in_cabinet` |
| `PickPlaceCounterToDrawer` | `pick_up_object(object, counter)` + `place_in_receptacle(object, drawer_interior)` | `pickup_from_counter` + `place_in_drawer` |
| `PickPlaceCounterToMicrowave` | `pick_up_object(object, counter)` + `place_in_receptacle(object, microwave_cavity)` | `pickup_from_counter` + `place_in_microwave` |
| `PickPlaceCounterToOven` | `pick_up_object(object, counter)` + `place_in_receptacle(object, oven_cavity)` | `pickup_from_counter` + `place_in_oven` |
| `PickPlaceCounterToSink` | `pick_up_object(object, counter)` + `place_in_receptacle(object, sink_basin)` | `pickup_from_counter` + `place_in_sink` |
| `PickPlaceCounterToStandMixer` | `pick_up_object(object, counter)` + `place_in_receptacle(object, stand_mixer_bowl)` | `pickup_from_counter` + `place_in_stand_mixer` |
| `PickPlaceCounterToStove` | `pick_up_object(object, counter)` + `place_on_surface(object, stove_burner)` | `pickup_from_counter` + `place_on_stove_burner` |
| `PickPlaceCounterToToasterOven` | `pick_up_object(object, counter)` + `place_in_receptacle(object, toaster_oven_cavity)` | `pickup_from_counter` + `place_in_toaster_oven` |
| `PickPlaceDrawerToCounter` | `pick_up_object(object, drawer_interior)` + `place_on_surface(object, counter)` | `pickup_from_drawer` + `place_on_counter` |
| `PickPlaceFridgeDrawerToShelf` | `pick_up_object(object, fridge_drawer)` + `place_on_surface(object, fridge_shelf)` | `pickup_from_fridge_drawer` + `place_on_fridge_shelf` |
| `PickPlaceFridgeShelfToDrawer` | `pick_up_object(object, fridge_shelf)` + `place_in_receptacle(object, fridge_drawer)` | `pickup_from_fridge_shelf` + `place_in_fridge_drawer` |
| `PickPlaceMicrowaveToCounter` | `pick_up_object(object, microwave_cavity)` + `place_on_surface(object, counter)` | `pickup_from_microwave` + `place_on_counter` |
| `PickPlaceSinkToCounter` | `pick_up_object(object, sink_basin)` + `place_on_surface(object, counter)` | `pickup_from_sink` + `place_on_counter` |
| `PickPlaceStoveToCounter` | `pick_up_object(object, stove_burner)` + `place_on_surface(object, counter)` | `pickup_from_stove_burner` + `place_on_counter` |
| `PickPlaceToasterOvenToCounter` | `pick_up_object(object, toaster_oven_cavity)` + `place_on_surface(object, counter)` | `pickup_from_toaster_oven` + `place_on_counter` |
| `PickPlaceToasterToCounter` | `pick_up_object(object, toaster_slot)` + `place_on_surface(object, counter)` | `pickup_from_toaster` + `place_on_counter` |
| `CheesyBread` | `pick_up_object(cheese, counter)` + `place_on_object(cheese, bread)` | `pickup_cheese` + `place_on_bread` |
| `MakeIcedCoffee` | `pick_up_object(ice_cube, bowl)` + `place_in_receptacle(ice_cube, cup)` | `pickup_ice_cube` + `place_in_cup` |
| `PackDessert` | `pick_up_object(dessert, counter)` + `place_in_receptacle(dessert, tupperware)` | `pickup_dessert` + `place_in_tupperware` |

### Access and Articulation Mapping

Most tasks in this section remain single subatomic tasks. Exceptions are noted implicitly
by their decomposition.

| Current RoboCasa atomic task | Canonical subatomic sequence | Suggested explicit alias |
| --- | --- | --- |
| `CloseBlenderLid` | `pick_up_object(blender_lid, counter)` + `place_on_surface(blender_lid, blender_lid_seat)` | `pickup_blender_lid` + `place_blender_lid_on_blender` |
| `CloseCabinet` | `close_hinged_part(cabinet, door)` | `close_cabinet_door` |
| `CloseDishwasher` | `close_hinged_part(dishwasher, door)` | `close_dishwasher_door` |
| `CloseDrawer` | `close_sliding_part(drawer, drawer)` | `close_drawer` |
| `CloseElectricKettleLid` | `close_hinged_part(electric_kettle, lid)` | `close_electric_kettle_lid` |
| `CloseFridge` | `close_hinged_part(fridge, door)` | `close_fridge_door` |
| `CloseFridgeDrawer` | `close_sliding_part(fridge, drawer)` | `close_fridge_drawer` |
| `CloseMicrowave` | `close_hinged_part(microwave, door)` | `close_microwave_door` |
| `CloseOven` | `close_hinged_part(oven, door)` | `close_oven_door` |
| `CloseStandMixerHead` | `close_hinged_part(stand_mixer, head)` | `close_stand_mixer_head` |
| `CloseToasterOvenDoor` | `close_hinged_part(toaster_oven, door)` | `close_toaster_oven_door` |
| `OpenBlenderLid` | `pick_up_object(blender_lid, blender_lid_seat)` + `place_on_surface(blender_lid, counter)` | `pickup_blender_lid` + `place_on_counter` |
| `OpenCabinet` | `open_hinged_part(cabinet, door)` | `open_cabinet_door` |
| `OpenDishwasher` | `open_hinged_part(dishwasher, door)` | `open_dishwasher_door` |
| `OpenDrawer` | `open_sliding_part(drawer, drawer)` | `open_drawer` |
| `OpenElectricKettleLid` | `press_button(electric_kettle, lid_release_button)` | `press_kettle_lid_release` |
| `OpenFridge` | `open_hinged_part(fridge, door)` | `open_fridge_door` |
| `OpenFridgeDrawer` | `open_sliding_part(fridge, drawer)` | `open_fridge_drawer` |
| `OpenMicrowave` | `open_hinged_part(microwave, door)` | `open_microwave_door` |
| `OpenOven` | `open_hinged_part(oven, door)` | `open_oven_door` |
| `OpenStandMixerHead` | `open_hinged_part(stand_mixer, head)` | `open_stand_mixer_head` |
| `OpenToasterOvenDoor` | `open_hinged_part(toaster_oven, door)` | `open_toaster_oven_door` |
| `SlideDishwasherRack` | `open_sliding_part(dishwasher, top_rack)` or `close_sliding_part(dishwasher, top_rack)` | `pull_out_dishwasher_rack` / `push_in_dishwasher_rack` |
| `SlideOvenRack` | `open_sliding_part(oven, rack)` or `close_sliding_part(oven, rack)` | `pull_out_oven_rack` / `push_in_oven_rack` |
| `SlideToasterOvenRack` | `open_sliding_part(toaster_oven, rack_or_tray)` or `close_sliding_part(toaster_oven, rack_or_tray)` | `pull_out_toaster_oven_rack` / `push_in_toaster_oven_rack` |

### Controls and Activation Mapping

Tasks that currently infer direction from state should become explicit goal-directed
aliases in the canonical layer.

| Current RoboCasa atomic task | Canonical subatomic sequence | Suggested explicit alias |
| --- | --- | --- |
| `AdjustToasterOvenTemperature` | `set_rotary_control(toaster_oven, temperature_knob, goal={increase,decrease})` | `increase_toaster_oven_temperature` / `decrease_toaster_oven_temperature` |
| `AdjustWaterTemperature` | `set_rotary_control(sink, faucet_handle, goal={hot,cold})` | `set_sink_water_hot` / `set_sink_water_cold` |
| `LowerHeat` | `set_rotary_control(stove, burner_knob, goal=low)` | `turn_stove_burner_low` |
| `PreheatOven` | `set_rotary_control(oven, temperature_knob, goal=preheat)` | `preheat_oven` |
| `StartCoffeeMachine` | `press_button(coffee_machine, start_button)` | `press_coffee_machine_button` |
| `TurnOffMicrowave` | `press_button(microwave, stop_button)` | `press_microwave_stop` |
| `TurnOffSinkFaucet` | `set_rotary_control(sink, faucet_handle, goal=water_off)` | `turn_sink_faucet_off` |
| `TurnOffStove` | `set_rotary_control(stove, burner_knob, goal=off)` | `turn_stove_burner_off` |
| `TurnOnBlender` | `press_button(blender, power_button)` | `press_blender_power_button` |
| `TurnOnElectricKettle` | `press_lever(electric_kettle, power_lever)` | `press_electric_kettle_power_lever` |
| `TurnOnMicrowave` | `press_button(microwave, start_button)` | `press_microwave_start` |
| `TurnOnSinkFaucet` | `set_rotary_control(sink, faucet_handle, goal=water_on)` | `turn_sink_faucet_on` |
| `TurnOnStove` | `set_rotary_control(stove, burner_knob, goal=on)` | `turn_stove_burner_on` |
| `TurnOnToaster` | `press_lever(toaster, toaster_lever)` | `press_toaster_lever` |
| `TurnOnToasterOven` | `set_rotary_control(toaster_oven, timer_knob, goal=on)` | `turn_toaster_oven_on` |
| `TurnSinkSpout` | `set_rotary_control(sink, spout, goal={left,right})` | `turn_sink_spout_left` / `turn_sink_spout_right` |

### Recommended Refactor Outcome

If RoboCasa is refactored around this taxonomy, the canonical subatomic inventory would
be 13 subatomic tasks:

1. `navigate_to_fixture`
2. `pick_up_object`
3. `place_on_surface`
4. `place_in_receptacle`
5. `place_on_object`
6. `place_under_dispenser`
7. `open_hinged_part`
8. `close_hinged_part`
9. `open_sliding_part`
10. `close_sliding_part`
11. `press_button`
12. `press_lever`
13. `set_rotary_control`

That canonical set is expressive enough to cover the current 65 RoboCasa atomic tasks
while making transport, access, and control semantics much easier to compose explicitly.
