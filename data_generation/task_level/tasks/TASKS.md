# Next Task Candidates

This file tracks the next RoboCasa composite tasks that look like good fits for
the task-level symbolic generation stack after `HotDogSetup`,
`PrepareCoffee`, and `PrepareSandwichStation`.

## Recommended 7

### 1. PrepareSausageCheese
Source: `robocasa/environments/kitchen/composite/preparing_sandwiches/prepare_sausage_cheese.py`

Why it is a good fit:
- Two-agent split is natural: one agent can fetch sausage while the other fetches cheese.
- Goal semantics are simple: both items end up on the cutting board.
- Very close in structure to the existing sandwich tasks.

### 2. PrepareCheeseStation
Source: `robocasa/environments/kitchen/composite/making_salads/prepare_cheese_station.py`

Why it is a good fit:
- Cross-fixture coordination: cheese from fridge, grater from cabinet.
- Final placement is easy to state symbolically: stage both near the bowl.
- Good communication potential without requiring new control primitives.

### 3. GatherMarinadeIngredients
Source: `robocasa/environments/kitchen/composite/preparing_marinade/gather_marinade_ingredients.py`

Why it is a good fit:
- Strongest coordination profile of the current shortlist.
- One agent can handle cabinet items while the other handles garlic from the fridge.
- Mix of `place_in_receptacle` and `place_next_to` style goals.

### 4. SetUpSpiceStation
Source: `robocasa/environments/kitchen/composite/seasoning_food/setup_spice_station.py`

Why it is a good fit:
- Three movable items makes the task nontrivial for two agents.
- Staging items near the stove aligns well with the existing symbolic abstractions.
- Minimal need for new simulator controls or task-local machinery.

### 5. OrganizeCondiments
Source: `robocasa/environments/kitchen/composite/arranging_condiments/organize_condiments.py`

Why it is a good fit:
- Good two-agent division of labor over three target condiments.
- Includes a distractor, which raises the semantic bar.
- Cabinet placement is already represented in the task-level tool vocabulary.

### 6. SetupFruitBowl
Source: `robocasa/environments/kitchen/composite/setting_the_table/setup_fruit_bowl.py`

Why it is a good fit:
- Simple cooperative split: one fruit per agent.
- Bowl placement semantics are already familiar to the symbolic validator.
- Good candidate for early scaling because it is easy to test.

### 7. OrganizeCoffeeCondiments
Source: `robocasa/environments/kitchen/composite/brewing/organize_coffee_condiments.py`

Why it is a good fit:
- Thematically close to `PrepareCoffee`.
- Good cross-fixture staging task without requiring cooking state.
- Easy prompt story: retrieve items from cabinet and stage them near the mug.

## Coordination-Heavy But Harder

### SetupBowls
Source: `robocasa/environments/kitchen/composite/setting_the_table/setup_bowls.py`

Why it is compelling:
- Excellent two-agent coordination and communication task.
- Clean work split: one bowl per agent.

Why it is harder:
- Needs a symbolic notion like `front_of(stool)` rather than only
  `on_surface`, `in_receptacle`, or `next_to`.

### SetupWineGlasses
Source: `robocasa/environments/kitchen/composite/setting_the_table/setup_wine_glasses.py`

Why it is compelling:
- Also naturally parallel and communication-heavy.

Why it is harder:
- Needs stable symbolic reasoning about left/right or side-of-plate geometry.

### AlignSilverware
Source: `robocasa/environments/kitchen/composite/setting_the_table/align_silverware.py`

Why it is compelling:
- Strong communication and ordering signal.

Why it is harder:
- Requires symbolic `left_of` / `right_of` relations relative to a plate.
