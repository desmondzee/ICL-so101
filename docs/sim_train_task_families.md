# Simulated training task families

Catalog of candidate skill families for `sim_train_v1` (target 30-40 qualified families, minimum 20, about 100 task definitions, about 10 accepted pairs per task). Each family is authored by one agent as `sim/train/tasks/families/<family>.py` with 2-5 task definitions that differ in objects, receptacles, relations or step order. A colour or layout swap is episode variation, not a new task. Every task must qualify (>= 48/50 strict physics passes) before production; families that cannot be made reliable are dropped or replaced, and the drop is recorded here.

Held out (validation, never in training): put the red/blue blocks on matching plates; white bowl in black bowl; red mug on a named plate; two mugs into the microwave; frying pan on the stove. Families below may share a manipulation family with these (recorded as family overlap) but never their exact semantics.

Assets: LIBERO packaged foods (alphabet_soup, tomato_sauce, cream_cheese, butter, chocolate_pudding, popcorn, ketchup, milk, orange_juice, bbq_sauce, cookies, macaroni_and_cheese, salad dressings), bowls (white, akita black, red), plate, ramekin, basket, frypan, simple_rack, mugs (porcelain, red_coffee, white_yellow), black/yellow books, desk_caddy, wooden_tray, white_storage_box, wooden_shelf, wooden_two_layer_shelf, wine_bottle/wine_rack, moka_pot, bowl_drainer; articulated short_cabinet, wooden_cabinet, white_cabinet (drawers/doors), short_fridge, basin_faucet, microwave, flat_stove (button); YCB banana, fork, spoon, knife, spatula, scissors, large marker, screwdriver, golf ball, gelatin box; GSO tape rolls, sponge pack, bottles, candy box, can opener, honey dipper, C-clamp, sprinkles canister; primitives (cube, cylinder, sphere, pyramid, disc/coaster/mat). Arenas: kitchen, living room, study, three tabletop styles, coffee-table and floor styles (qualified by workstream A).

| # | Family | Example task definitions | Skills | Risk |
|---|---|---|---|---|
| 1 | put_in_container | soup can into basket; butter into storage box; golf ball into ramekin; marker into desk caddy | pick, place-in | low |
| 2 | take_out_of_container | cream cheese out of basket onto mat; banana out of bowl onto plate | pick from container, place | low |
| 3 | place_on_surface | banana on plate; tomato sauce on wooden tray; sponge on coaster | pick, place-on | low |
| 4 | place_on_elevated | box on top of book; can on top of storage box | pick, place high | low |
| 5 | place_on_shelf | carton onto wooden shelf; can onto upper level of two-layer shelf | pick, reach high, place | medium (reach) |
| 6 | relative_left_right | put the pudding to the left of the bowl; the butter to the right of the mug | pick, relational place | low |
| 7 | relative_front_behind | put the can in front of the basket; the box behind the plate | pick, relational place | low |
| 8 | place_between | put the ball between the two cans; the block between two plates | pick, place-between | low |
| 9 | stack_two | cream cheese box on butter; cube on cube; candy box on gelatin box | pick, stack | low |
| 10 | stack_three_ordered | three cubes in a named order; three food boxes | ordered stack | medium |
| 11 | unstack | take the top box off and put it on the table/mat | pick from stack, place | low |
| 12 | nest_containers | ramekin into a bowl; small bowl into the basket | rim grasp, place-in | medium |
| 13 | gather_two | put the soup and the sauce into the basket (in order); two balls into a bowl | two ordered pick-places | low |
| 14 | sort_by_category | food boxes into the basket and utensils into the caddy | multi-object, two targets | medium |
| 15 | transfer_between_containers | move the ball from the bowl to the ramekin; the box from the tray to the basket | pick from container, place-in | low |
| 16 | empty_container | take both items out of the basket onto the table | two ordered removals | medium |
| 17 | line_up_row | line up three cans along the mat edge | ordered placements in a row | medium |
| 18 | swap_positions | swap the soup can and the sauce bottle | three moves via a free spot | medium |
| 19 | center_on_target | centre the box on the coaster; the mug on the mat centre | precise place | low |
| 20 | reorient_yaw | turn the book/long block crosswise; align a box with the tray | pick, rotate in air, place | medium |
| 21 | lay_down | lay a standing bottle on its side | pick, rotate 90 deg, place | medium |
| 22 | stand_upright | stand a lying can upright on its base | pick, rotate 90 deg, place | high |
| 23 | push_to_target | push the cube into the marked square; slide the box onto the mat | non-prehensile push | medium |
| 24 | push_together | push two blocks until they touch | pushes | medium |
| 25 | utensil_to_holder | fork into the caddy; marker into the mug; spoon into the bowl | thin-object grasp, place-in | medium |
| 26 | put_on_plate_in_order | banana then ball onto the plate (named order) | two ordered placements | low |
| 27 | drawer_open | pull the cabinet drawer open | handle grasp, linear pull | high |
| 28 | drawer_close | push the open drawer closed | push along axis | medium |
| 29 | put_in_drawer | put the box into the open drawer | pick, place-in (low clearance) | high |
| 30 | door_close | close the fridge/cabinet door; close the microwave door | push/swing | medium |
| 31 | door_open | open the cabinet door by its edge | pull/swing | high |
| 32 | press_button | press the stove button | press | medium |
| 33 | turn_faucet | turn the faucet handle | rotate articulated handle | high |
| 34 | place_in_rack | put the plate/bowl into the dish rack | orientation-sensitive insert | high |
| 35 | bottle_in_wine_rack | put the bottle into the wine rack slot | orientation insert | high |
| 36 | cover_object | put the bowl upside down over the ball | flip grasp | high |
| 37 | tool_to_tray | put the screwdriver/scissors on the tray | thin-object grasp | medium |
| 38 | moka_pot_on_coaster | put the moka pot on the coaster | handle-aware place | medium |

Allocation rule: author all low/medium families first, then high-risk ones in parallel as time allows; a family must reach at least two qualified task definitions to count. Target count follows qualification results; the release card reports actual families and tasks.
