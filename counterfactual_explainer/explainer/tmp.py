"""
0: bathroom
1: shower
2: bathroom_cabinet
3: bathroom_counter
4: dining_room
5: knifeblock
6: mat
7: bench
8: table
9: cupboard
10: kitchen_counter
11: sink
12: dishwasher
13: bedroom
14: bookshelf
15: home_office
16: sofa
17: table
18: desk
19: tvstand
20: outside
21: stove
22: fridge
23: dresser
24: shoe_rack
25: bag
26: book
27: cd
28: cd_player
29: coffee
30: coffee_cup
31: conditioner
32: cookingpot
33: drinking_glass
34: facial_cleanser
35: food_jam
36: food_peanut_butter
37: food_rice
38: food_vegetable
39: hairbrush
40: hairdryer
41: instrument_guitar
42: keys
43: knife
44: mail
45: mug
46: note_pad
47: oil
48: painkillers
49: pencil
50: plate
51: remote_control
52: shampoo
53: shoes
54: spectacles
55: spoon
56: tea
57: tooth_paste
58: toothbrush
59: towel
60: towel_rack
61: trashcan
62: bookshelf
63: sofa
64: bookshelf
65: kitchen_cabinet
66: bowl
67: chessboard
68: cutting_board
69: deck_of_cards
70: food_cereal
71: food_cheese
72: food_donut
73: headset
74: highlighter
75: milk
76: notebook
77: pen
78: trashbag
79: vacuum_cleaner
80: wine
81: wine_glass
82: washing_machine
83: basket_for_clothes
84: cloth_napkin
85: drying_rack
86: keyboard
87: laundry_detergent
88: napkin
89: kitchen_counter
90: coffe_maker
91: coffee_filter
92: dishrack
93: dishtowel
94: face_soap
95: food_apple
96: food_bread
97: food_egg
98: fork
99: groceries
100: ground_coffee
101: pajamas
102: sauce_pan
103: toothbrush_holder
104: dish_soap
105: instrument_violin
106: radio
107: sponge
"""
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','helpers')))


from stot_model import STOTModel
from data_utils import DatasetManager
from movement_tracker import MovementTracker
from perturbation import PerturbationEngine
from logger import Logger  
from debugger import Debugger  






class Explainer:
    def __init__(self, step_size):

        # Initialize core components
        self.model = STOTModel(step_size=step_size)
        # self.logger = Logger()

        # Load dataset
        self.data_manager = DatasetManager(
            time_encoder=self.model.time_encoder,
            batch_size=self.model.model_configs['batch_size'],
            train_days=30
        )

        self.num_nodes = self.model.num_nodes
        self.movement_tracker = MovementTracker(num_nodes=self.num_nodes)
        self.debugger = Debugger(self.data_manager.dataset.node_classes, self.movement_tracker)
        
        self.perturb_engine = PerturbationEngine(
            model=self.model,
            movement_tracker=self.movement_tracker,
            debugger=self.debugger
        )
        self.step_size = step_size

    def run(self):
        """
        Main inference + counterfactual analysis loop.
        Loads test routines, detects movement, and performs perturbation tests.
        """
        test_routines = self.data_manager.test_routines
        time_step_size = []
        for day_no,(day_routine, additional_info) in enumerate(test_routines):
            print(f"Processing day {day_no + 1}/{len(test_routines)}...")
            routine_iterator = self.data_manager.get_iterator(day_routine, step_size=self.step_size)
            self.movement_tracker.reset()  # Reset movement tracker for each day 
            
            for no, routine_window in enumerate(routine_iterator):
                print(f"Processing routine {no + 1}/{len(day_routine) - self.step_size + 1}...")
                time = routine_window[0]['time']
                time2 = routine_window[1]['time']
                diff = time2 - time
                if diff not in time_step_size:
                    time_step_size.append(diff)
                print(f"Time: {time}")
                # Step 1: Inference:
                # inp, pred, gt, edge_probs = self.model.infer(routine_window)
                # # self.debugger.verify_model_returns(inp, gt, routine_window)
                # self.debugger.visualize_model_run(inp, gt, pred)
                # self.debugger.pretty_print_movement_detected()
                # # # Step 2: Peturbation:
                # explanation_results = self.perturb_engine.run(routine_window, inp, pred)
                # # # Step 3: log explanation results
                # # self.logger.log_explanation(day_no, no, explanation_results)
                # # # Step 4: Movement tracking
                # self.movement_tracker.update(routine_window)
        
        print(f"Unique time step sizes: {time_step_size}")
            





if __name__ == "__main__":
    explainer = Explainer(step_size=2)
    explainer.run()
    print("Explainer run completed.")