# Verified TaskSpecs

This directory stores the canonical verified TaskSpec inventory.

- One JSON file per task slug
- No duplicate group-local copies
- File names remain the normalized task slug, for example
  `preparecoffee.json` or `colorfulsalsa.json`

Behavior-family group membership is preserved:

- programmatically in `data_generation/task_level/pipeline/task_groups.py`
- for humans in the inventory table below

When older verified trees contained duplicate variants for the same task, the
canonical file here keeps one working spec per task and the inventory below
records every original source path.

## Inventory

| Task Slug | Composite Task | Canonical Source | Source Groups | Source Spec Paths | Notes |
| --- | --- | --- | --- | --- | --- |
| `addlemontofish` | `AddLemonToFish` | `garnishes/addlemontofish.json` | garnishes<br>meat_pan_serving | garnishes/addlemontofish.json<br>meat_pan_serving/addlemontofish.json | kept working variant; dropped failing duplicate(s) |
| `addsugarcubes` | `AddSugarCubes` | `plates/addsugarcubes.json` | plates | plates/addsugarcubes.json | canonical |
| `alcoholservingprep` | `AlcoholServingPrep` | `beverages_drinkware/alcoholservingprep.json` | beverages_drinkware | beverages_drinkware/alcoholservingprep.json | canonical |
| `alignsilverware` | `AlignSilverware` | `silverware/alignsilverware.json` | silverware | silverware/alignsilverware.json | canonical |
| `arrangebreadbowl` | `ArrangeBreadBowl` | `bowls/arrangebreadbowl.json` | bowls | bowls/arrangebreadbowl.json | canonical |
| `arrangedrinkware` | `ArrangeDrinkware` | `beverages_drinkware/arrangedrinkware.json` | beverages_drinkware | beverages_drinkware/arrangedrinkware.json | canonical |
| `arrangetea` | `ArrangeTea` | `beverages_drinkware/arrangetea.json` | beverages_drinkware | beverages_drinkware/arrangetea.json | canonical |
| `beverageorganization` | `BeverageOrganization` | `beverages_drinkware/beverageorganization.json` | beverages_drinkware | beverages_drinkware/beverageorganization.json | canonical |
| `bowlandcup` | `BowlAndCup` | `bowls/bowlandcup.json` | bowls | bowls/bowlandcup.json | canonical |
| `candlecleanup` | `CandleCleanup` | `sink_cleaning/candlecleanup.json` | sink_cleaning | sink_cleaning/candlecleanup.json | canonical |
| `cheesemixing` | `CheeseMixing` | `blender_mixer/cheesemixing.json` | blender_mixer | blender_mixer/cheesemixing.json | canonical |
| `clusteritemsforclearing` | `ClusterItemsForClearing` | `clearing_clustering/clusteritemsforclearing.json` | clearing_clustering | clearing_clustering/clusteritemsforclearing.json | canonical |
| `colorfulsalsa` | `ColorfulSalsa` | `blender_mixer/colorfulsalsa.json` | blender_mixer | blender_mixer/colorfulsalsa.json | canonical |
| `condimentcollection` | `CondimentCollection` | `condiments_spices/condimentcollection.json` | condiments_spices | condiments_spices/condimentcollection.json | canonical |
| `cutbuffetpizza` | `CutBuffetPizza` | `buffet_station_setup/cutbuffetpizza.json` | buffet_station_setup | buffet_station_setup/cutbuffetpizza.json | canonical |
| `datenight` | `DateNight` | `beverages_drinkware/datenight.json` | beverages_drinkware | beverages_drinkware/datenight.json | canonical |
| `deliverstraw` | `DeliverStraw` | `beverages_drinkware/deliverstraw.json` | beverages_drinkware | beverages_drinkware/deliverstraw.json | canonical |
| `dessertassembly` | `DessertAssembly` | `trays/dessertassembly.json` | trays | trays/dessertassembly.json | canonical |
| `displaymeatvariety` | `DisplayMeatVariety` | `meat_pan_serving/displaymeatvariety.json` | meat_pan_serving<br>trays | meat_pan_serving/displaymeatvariety.json<br>trays/displaymeatvariety.json | deduplicated overlapping verified variants |
| `distributechicken` | `DistributeChicken` | `meat_pan_serving/distributechicken.json` | meat_pan_serving | meat_pan_serving/distributechicken.json | canonical |
| `garnishcake` | `GarnishCake` | `garnishes/garnishcake.json` | garnishes | garnishes/garnishcake.json | canonical |
| `garnishcupcake` | `GarnishCupcake` | `garnishes/garnishcupcake.json` | garnishes | garnishes/garnishcupcake.json | canonical |
| `garnishpancake` | `GarnishPancake` | `garnishes/garnishpancake.json` | garnishes | garnishes/garnishpancake.json | canonical |
| `gathermarinadeingredients` | `GatherMarinadeIngredients` | `condiments_spices/gathermarinadeingredients.json` | condiments_spices | condiments_spices/gathermarinadeingredients.json | canonical |
| `hotdogsetup` | `HotDogSetup` | `plates/hotdogsetup.json` | plates | plates/hotdogsetup.json | canonical |
| `lemonseasoningfish` | `LemonSeasoningFish` | `condiments_spices/lemonseasoningfish.json` | condiments_spices<br>garnishes<br>meat_pan_serving | condiments_spices/lemonseasoningfish.json<br>garnishes/lemonseasoningfish.json<br>meat_pan_serving/lemonseasoningfish.json | deduplicated overlapping verified variants |
| `lineupcondiments` | `LineUpCondiments` | `condiments_spices/lineupcondiments.json` | condiments_spices | condiments_spices/lineupcondiments.json | canonical |
| `matchcupanddrink` | `MatchCupAndDrink` | `beverages_drinkware/matchcupanddrink.json` | beverages_drinkware | beverages_drinkware/matchcupanddrink.json | canonical |
| `meatskewerassembly` | `MeatSkewerAssembly` | `meat_pan_serving/meatskewerassembly.json` | meat_pan_serving | meat_pan_serving/meatskewerassembly.json | canonical |
| `organizecondiments` | `OrganizeCondiments` | `condiments_spices/organizecondiments.json` | condiments_spices | condiments_spices/organizecondiments.json | canonical |
| `platestoredinner` | `PlateStoreDinner` | `plates/platestoredinner.json` | plates | plates/platestoredinner.json | canonical |
| `portionhotdogs` | `PortionHotDogs` | `bowls/portionhotdogs.json` | bowls<br>plates | bowls/portionhotdogs.json<br>plates/portionhotdogs.json | deduplicated overlapping verified variants |
| `portionyogurt` | `PortionYogurt` | `bowls/portionyogurt.json` | bowls<br>plates | bowls/portionyogurt.json<br>plates/portionyogurt.json | deduplicated overlapping verified variants |
| `preparecheesestation` | `PrepareCheeseStation` | `buffet_station_setup/preparecheesestation.json` | buffet_station_setup | buffet_station_setup/preparecheesestation.json | canonical |
| `preparecocktailstation` | `PrepareCocktailStation` | `beverages_drinkware/preparecocktailstation.json` | beverages_drinkware | beverages_drinkware/preparecocktailstation.json | canonical |
| `preparecoffee` | `PrepareCoffee` | `beverages_drinkware/preparecoffee.json` | beverages_drinkware | beverages_drinkware/preparecoffee.json | canonical |
| `preparedrinkstation` | `PrepareDrinkStation` | `beverages_drinkware/preparedrinkstation.json` | beverages_drinkware | beverages_drinkware/preparedrinkstation.json | canonical |
| `preparesausagecheese` | `PrepareSausageCheese` | `meat_pan_serving/preparesausagecheese.json` | meat_pan_serving<br>plates | meat_pan_serving/preparesausagecheese.json<br>plates/preparesausagecheese.json | deduplicated overlapping verified variants |
| `preparesoupserving` | `PrepareSoupServing` | `buffet_station_setup/preparesoupserving.json` | buffet_station_setup | buffet_station_setup/preparesoupserving.json | canonical |
| `seasoningsteak` | `SeasoningSteak` | `condiments_spices/seasoningsteak.json` | condiments_spices<br>meat_pan_serving | condiments_spices/seasoningsteak.json<br>meat_pan_serving/seasoningsteak.json | deduplicated overlapping verified variants |
| `servemealjuice` | `ServeMealJuice` | `beverages_drinkware/servemealjuice.json` | beverages_drinkware | beverages_drinkware/servemealjuice.json | canonical |
| `servesteak` | `ServeSteak` | `meat_pan_serving/servesteak.json` | meat_pan_serving<br>plates | meat_pan_serving/servesteak.json<br>plates/servesteak.json | deduplicated overlapping verified variants |
| `setbowlsforsoup` | `SetBowlsForSoup` | `silverware/setbowlsforsoup.json` | silverware | silverware/setbowlsforsoup.json | canonical |
| `setupspicestation` | `SetUpSpiceStation` | `condiments_spices/setupspicestation.json` | condiments_spices | condiments_spices/setupspicestation.json | canonical |
| `setupbowls` | `SetupBowls` | `silverware/setupbowls.json` | silverware | silverware/setupbowls.json | canonical |
| `setupbutterplate` | `SetupButterPlate` | `plates/setupbutterplate.json` | plates | plates/setupbutterplate.json | canonical |
| `setupsodabowl` | `SetupSodaBowl` | `beverages_drinkware/setupsodabowl.json` | beverages_drinkware | beverages_drinkware/setupsodabowl.json | canonical |
| `setupwineglasses` | `SetupWineGlasses` | `beverages_drinkware/setupwineglasses.json` | beverages_drinkware | beverages_drinkware/setupwineglasses.json | canonical |
| `spicymarinade` | `SpicyMarinade` | `condiments_spices/spicymarinade.json` | condiments_spices | condiments_spices/spicymarinade.json | canonical |
| `sweetencoffee` | `SweetenCoffee` | `beverages_drinkware/sweetencoffee.json` | beverages_drinkware | beverages_drinkware/sweetencoffee.json | canonical |
| `tongbuffetsetup` | `TongBuffetSetup` | `buffet_station_setup/tongbuffetsetup.json` | buffet_station_setup | buffet_station_setup/tongbuffetsetup.json | canonical |
| `veggiedipprep` | `VeggieDipPrep` | `bowls/veggiedipprep.json` | bowls<br>trays | bowls/veggiedipprep.json<br>trays/veggiedipprep.json | deduplicated overlapping verified variants |
