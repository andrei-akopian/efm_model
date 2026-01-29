import numpy as np
import json
import argparse
import sys

# --- Configuration & Constants ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SIM_DURATION = 80.0
FPS = 20
DT = 1 / FPS
TOTAL_STEPS = int(SIM_DURATION / DT)

# Schedule of sex hormone change throughout the simulation
T1 = 20 * FPS
T2 = T1 + 50 * FPS
SCHEDULE = [ # full program
    (0, T1, 1),
    (T1, T2, 2),
    (T2, TOTAL_STEPS, 0)
]
# SCHEDULE = [ # equilibrium only
#     (0, TOTAL_STEPS, 1),
# ]
# SCHEDULE = [ # growth only
#     (0, TOTAL_STEPS, 2),
# ]
# SCHEDULE = [ # decline only
#     (0, TOTAL_STEPS, 0),
# ]

SH_COLORS = {0: '#ff9999', 1: '#d3d3d3', 2: '#90ee90'}

# --- Cell Constants ---
# EC (Rigid)
E_MAX_WIDTH, E_HEIGHT, E_GROWTH_RATE = 1.0, 1.0, 0.5
E_DIV_PROB = 0.15 # does nothing
E_LOBEC_PROB = 0.25
E_DEATH_PROB = 0.036 # probability per second

# Elastic Cells (F & M)
# To restrain fibs and macs to EC length,
# treat them as springs where mass is "ability to stretch"
F_MAX_MASS = 12.0
F_HEIGHT, F_GROWTH_RATE = 0.5, 0.14
F_MIN_DENSITY = 1.0
F_MAX_WIDTH = F_MAX_MASS
F_DIV_PROB = 0.4
F_DEATH_PROB = 0.05 # probability per second

M_MAX_MASS = 12.0
M_HEIGHT, M_GROWTH_RATE = 0.5, 1
M_MIN_DENSITY = 1.0
M_MAX_WIDTH = M_MAX_MASS
M_DIV_PROB = 0.3
M_DEATH_PROB = 0.05 # probability per second

MAC_EFFECT_ON_FIB = 4

FM_DIVISION_PERMISSION_THRESHOLD = 1 # at which width to begin divisions. Avoids sudden jumps.

ECM_SUPPLY_RATE = 0.2
COLORS = {
    'm': '#90EE90', 'f': '#FFB347', 'e': '#F5DEB3',
    'lob': '#D3D3D3', 'ecm': '#FFA500', 'dying': '#FFADAD'
}
ECM_HEIGHT = 0.2

# ---- Behaviors ----
F_LIMIT_E = True
MF_WIDTH_LIMITS = False
FAKE_MAC = False
FAKE_FIB = False
MAC_AFFECT_FIB = True
DEUS_EX_MACHINA = True # messes with "random" probabilities to keep ratios to the ideal

# --- Class Definitions ---
class Cell:
    SH: int = 1 # constant
    def __init__(self, initial_val, max_val, height, color, growth_rate, div_prob, death_prob, is_elastic=False):
        self.age = 0.0
        self.height = height
        self.base_color = color
        self.base_growth_rate = growth_rate
        self.base_div_prob = div_prob
        self.base_death_prob = death_prob
        self.left_edge = 0.0
        self.right_edge = 0.0
        self.dying = False

        self.is_elastic = is_elastic
        if is_elastic:
            self.mass = initial_val
            self.max_mass = max_val
            self.width = 0
            self.max_width = 999.0
            self.min_density = 1.0
        else:
            self.width = initial_val
            self.max_width = max_val

    def get_pos_info(self):
        return {
            'left': self.left_edge, 'right': self.right_edge,
            'width': self.width,
            'color': COLORS['dying'] if self.dying else self.base_color,
            'height': self.height, 'age': self.age,
            'dying': self.dying
        }

    def update(self, multiplier=1.0):
        # handeled by the subclasses individually
        pass

    def calculate_death(self, multiplier=1.0) -> bool | None:
        # handeled by the subclasses individually
        pass

    def calculate_division(self, multiplier=1.0):
        # special cases
        if self.dying:
            return False
        if self.width < (0.5 * self.max_width):
            return False
        # put a function in terms of width or age here is necessary
        return self.width >= self.max_width

class EC(Cell):
    ec_cell_count = 0
    def __init__(self, initial_width=0.4, cell_type="EC"):
        color = COLORS['lob'] if cell_type == "LobEC" else COLORS['e']
        super().__init__(initial_width, E_MAX_WIDTH, E_HEIGHT, color, E_GROWTH_RATE, E_DIV_PROB, E_DEATH_PROB, is_elastic=False)
        # self.ecm_concentration = 0.5 # currently unused
        self.cell_type = cell_type
        # self.covered_by_fib = False
        self.fibroblast_density = 0.0

    def update(self, multiplier=1.0):
        self.age += DT
        change = self.base_growth_rate * DT * multiplier
        # growth = change * multiplier * self.covered_by_fib
        if self.dying:
            self.width -= change * 2
        else:
            if F_LIMIT_E and self.fibroblast_density<1:
                change *= (self.fibroblast_density / EC.ec_cell_count)
                # growth = 0
            growth = change * [0,0.1,2][Cell.SH]
            if self.width < self.max_width:
                self.width = min(self.max_width, self.width + growth)

    def calculate_division(self, multiplier=1.0):
        if self.cell_type == "LobEC":
            return False
        # if self.ecm_concentration < 1:
        #     return False
        # multiplier += self.ecm_concentration
        return super().calculate_division(multiplier)

    type_determiner = 0
    @staticmethod
    def get_next_type():
        types = ["EC", "LobEC"]
        if DEUS_EX_MACHINA:
            EC.type_determiner += E_LOBEC_PROB
            if EC.type_determiner >= 1:
                EC.type_determiner %= 1
                return types[1]
            else:
                return types[0]
        else:
            type_probabilities = [1 - E_LOBEC_PROB, E_LOBEC_PROB]
            return np.random.choice(types, p=type_probabilities)

    def divide(self):
        # set child constants
        child_w = self.width / 2
        d1_type = EC.get_next_type()
        d2_type = EC.get_next_type()
        d1 = EC(child_w, cell_type=d1_type)
        d2 = EC(child_w, cell_type=d2_type)
        # child_ecm = self.ecm_concentration / 2
        # d1.ecm_concentration = child_ecm
        # d2.ecm_concentration = child_ecm
        d1.left_edge, d1.right_edge = self.left_edge, self.left_edge + child_w
        d2.left_edge, d2.right_edge = d1.right_edge, self.right_edge
        return d1, d2

    death_determiner = 0
    @staticmethod
    def death_roll(celltype, probability):
        if DEUS_EX_MACHINA:
            celltype.death_determiner += probability
            if celltype.death_determiner >= 1:
                celltype.death_determiner %= 1
                return True
            else:
                return False
        else:
            if np.random.random() < probability:
                return True
            else:
                return False

    def calculate_death(self, multiplier=1.0):
        if self.dying:
            return False
        elif not self.dying:
            prob = self.base_death_prob * [2,1,1][Cell.SH] * multiplier * DT
            outcome  = EC.death_roll(EC, prob)
            self.dying = outcome
            return outcome

    def get_pos_info(self):
        info = super().get_pos_info()
        # info['ecm'] = self.ecm_concentration
        info['type'] = self.cell_type
        return info

class ElasticCell(Cell):
    def __init__(self, initial_mass, max_mass, color, height, growth_rate, div_prob, death_prob, min_density, max_width_limit):
        super().__init__(initial_mass, max_mass, height, color, growth_rate, div_prob, death_prob, is_elastic=True)
        self.min_density = min_density
        self.max_width = max_width_limit

    def update(self, multiplier = 1.0):
        self.age += DT
        change = self.base_growth_rate * DT
        growth = change * multiplier
        if self.dying:
            self.mass -= change * 5
        elif self.mass < self.max_mass:
            self.mass = min(self.max_mass, self.mass + growth)

    @staticmethod
    def division_roll(celltype, probability):
        if DEUS_EX_MACHINA:
            celltype.division_determiner += probability
            if celltype.division_determiner >= 1:
                celltype.division_determiner %= 1
                return True
            else:
                return False
        else:
            return np.random.random() < probability

    def calculate_division(self, multiplier=1.0):
        if self.dying:
            return False
        if self.width < (0.55 * self.max_width):
            return False
        elif (self.width >= (self.max_width * FM_DIVISION_PERMISSION_THRESHOLD)) and (self.mass == self.max_mass):
            probability = self.base_div_prob * DT
            return ElasticCell.division_roll(type(self), probability)
        return False

    def divide(self):
        child_mass = self.mass / 2
        d1 = type(self)(child_mass)
        d2 = type(self)(child_mass)
        return d1, d2

    @staticmethod
    def death_roll(celltype, probability):
        """
        celltype, refers to the child cell types, so their individual death determiner can be used.
        """
        if DEUS_EX_MACHINA:
            celltype.death_determiner += probability
            if celltype.death_determiner >= 1:
                celltype.death_determiner %= 1
                return True
            else:
                return False
        else:
            if np.random.random() < probability:
                return True
            else:
                return False

    def calculate_death(self, multiplier=1.0):
        if self.dying: # if cell already dead, don't interfere
            return False
        elif not self.dying:
            prob = self.base_death_prob * DT * [1,0,0][Cell.SH] * multiplier
            outcome = ElasticCell.death_roll(type(self), prob)
            self.dying = outcome
            return outcome

class Fibroblast(ElasticCell):
    # deus ex machina tracking variables
    death_determiner = 0
    division_determiner = 0
    def __init__(self, initial_mass=F_MAX_MASS/2):
        super().__init__(
            initial_mass=initial_mass,
            max_mass = F_MAX_MASS,
            color = COLORS['f'],
            height = F_HEIGHT,
            growth_rate = F_GROWTH_RATE,
            div_prob = F_DIV_PROB,
            death_prob = F_DEATH_PROB,
            min_density = F_MIN_DENSITY,
            max_width_limit= F_MAX_WIDTH
        )
        self.ec_border_count = 0
        self.has_mac = False
    def calculate_division(self, multiplier=1.0):
        return super().calculate_division(multiplier)
    def update(self, multiplier=1.0):
        multiplier = [0,0.3,1][Cell.SH] * multiplier
        if self.has_mac:
            multiplier *= MAC_EFFECT_ON_FIB
        super().update(multiplier)

class Macrophage(ElasticCell):
    # deus ex machina tracking variables
    death_determiner = 0
    division_determiner = 0
    def __init__(self, initial_mass=M_MAX_MASS/2):
        super().__init__(
            initial_mass=initial_mass,
            max_mass = M_MAX_MASS,
            color = COLORS['m'],
            height = M_HEIGHT,
            growth_rate = M_GROWTH_RATE,
            div_prob = M_DIV_PROB,
            death_prob = M_DEATH_PROB,
            min_density = M_MIN_DENSITY,
            max_width_limit= M_MAX_WIDTH
        )
        self.fib_border_count = 0
    def calculate_division(self, multiplier=1.0):
        return super().calculate_division(multiplier)

# --- Helper Functions ---
def set_rigid_edges(cells):
    total_w = sum(c.width for c in cells)
    curr_x = -total_w / 2
    for c in cells:
        c.left_edge, c.right_edge = curr_x, curr_x + c.width
        curr_x = c.right_edge
    return total_w

def distribute_elastic_cells(cells, available_width) -> float | None:
    count = len(cells)
    if count == 0:
        return None
    total_mass = sum(c.mass for c in cells)
    density = total_mass / available_width

    stretched_to_limits = False
    if MF_WIDTH_LIMITS:
        stretched_to_limits = density<=1

    for c in cells:
        if stretched_to_limits:
            c.width = c.mass
        else:
            c.width = c.mass / density

    curr_x = -available_width / 2
    for c in cells:
        c.left_edge = curr_x
        c.right_edge = curr_x + c.width
        curr_x = c.right_edge

    return density

def random_strict_initial_ec_types(p, N):
    n_LobEC = int(round(p * N))
    n_EC = N - n_LobEC
    arr = np.array(['EC'] * n_EC + ['LobEC'] * n_LobEC)
    np.random.shuffle(arr)
    return arr

# --- Simulation Logic ---
def run_simulation(initial_e, prop_lobec, initial_f, initial_m):
    # Initialize Lists using numpy random
    e_cell_types_distribution = random_strict_initial_ec_types(prop_lobec, initial_e)
    e_cells = [EC(np.random.uniform(0.5, 0.9), cell_type=e_cell_types_distribution[i]) for i in range(initial_e)]
    f_cells = [Fibroblast(np.random.uniform(0.25, 0.8)*F_MAX_MASS) for _ in range(initial_f)]
    m_cells = [Macrophage(np.random.uniform(0.25, 0.8)*M_MAX_MASS) for _ in range(initial_m)]
    history = []

    print(f"Simulating [E:{initial_e}, F:{initial_f}, M:{initial_m}] for {SIM_DURATION}s...")

    for step in range(TOTAL_STEPS):
        Cell.SH = 1
        for start, end, level in SCHEDULE:
            if start <= step < end:
                Cell.SH = level
                break

        # 1. Position ECs
        ec_total_width = set_rigid_edges(e_cells)

        # 2. Position Elastic Cells
        fibroblast_density = distribute_elastic_cells(f_cells, ec_total_width)
        distribute_elastic_cells(m_cells, ec_total_width)

        history.append({
            'e': [e.get_pos_info() for e in e_cells],
            'f': [f.get_pos_info() for f in f_cells],
            'm': [m.get_pos_info() for m in m_cells],
            'time': step * DT, 'sh': Cell.SH,
            'ec_total_width': ec_total_width
        })

        next_e, next_f, next_m = [], [], []

        # EC Update
        EC.ec_cell_count = len(e_cells)
        for e in e_cells:
            # e.covered_by_fib = any(not f.dying and f.right_edge > e.left_edge and f.left_edge < e.right_edge for f in f_cells)
            e.fibroblast_density = fibroblast_density

            e.update()
            e.calculate_death()

            if e.width > 0: # to remove shrunk dead cells
                if not e.dying and e.calculate_division():
                    next_e.extend(e.divide())
                else:
                    next_e.append(e)

        # Fibroblast Update
        enough_macs = len(m_cells) >= len(f_cells)
        for f in f_cells:
            f.ec_border_count = sum(1 for e in e_cells if not e.dying and e.right_edge > f.left_edge and e.left_edge < f.right_edge)

            # f.has_mac = FAKE_MAC or any(not m.dying and m.right_edge >= f.left_edge and m.left_edge <= f.right_edge for m in m_cells)
            if MAC_AFFECT_FIB:
                f.has_mac = FAKE_MAC or enough_macs
            else:
                f.has_mac = False

            f.update()
            f.calculate_death()

            if f.mass > 0:
                if f.calculate_division():
                    next_f.extend(f.divide())
                else:
                    next_f.append(f)

        # Macrophage Update
        # f_div_happened = len(next_f) > len(f_cells)
        # enough_macs = len(m_cells) >= len(f_cells)
        for m in m_cells:
            m.fib_border_count = sum(1 for f in f_cells if not f.dying and f.right_edge > m.left_edge and f.left_edge < m.right_edge)

            m.update()
            m.calculate_death()

            if m.mass > 0:
                # m_mult = 2.0 if f_div_happened else 0.5
                if m.calculate_division():
                    next_m.extend(m.divide())
                else:
                    next_m.append(m)

        e_cells, f_cells, m_cells = next_e, next_f, next_m

    print("\tFinished Simulating")
    return history

if __name__ == "__main__":

    # --- Defaults ---
    DEFAULT_E = 30
    DEFAULT_LOBEC = 0.5
    DEFAULT_F = 3
    DEFAULT_M = 3

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Cell Simulation")
    parser.add_argument("--e", type=int, default=DEFAULT_E, help=f"Initial Epithelial cells (default: {DEFAULT_E})")
    parser.add_argument("--lobec", type=float, default=DEFAULT_LOBEC, help=f"Proportion of LobEC cells (default: {DEFAULT_LOBEC})")
    parser.add_argument("--f", type=int, default=DEFAULT_F, help=f"Initial Fibroblasts (default: {DEFAULT_F})")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help=f"Initial Macrophages (default: {DEFAULT_M})")
    parser.add_argument("--default", action='store_true', help="Run default configuration")
    parser.add_argument("--seed", type=int, default=None, help="set a specific seed")

    # Check if CLI arguments are provided (length > 1 means script name + args)
    if len(sys.argv) > 1:
        print("=== CLI Simulation Setup ===")
        args = parser.parse_args()
        initial_e = args.e
        prop_lobec = args.lobec
        initial_f = args.f
        initial_m = args.m
        print(f"Parameters: E={initial_e}, LobEC={prop_lobec}, F={initial_f}, M={initial_m}")
        if args.seed is not None:
            if DEUS_EX_MACHINA:
                print("dues_ex_machina mode active, setting --seed has little effect")
            np.random.seed(args.seed)
    else:
        # Interactive Mode
        print("=== Cell Simulation Setup ===")
        try:
            in_e = input(f"Enter initial Epithelial cells [default {DEFAULT_E}]: ")
            initial_e = int(in_e) if in_e.strip() else DEFAULT_E

            in_prop_lobec = input(f"Proportion of LobEC cells [default {DEFAULT_LOBEC}]: ")
            # Fixed bug: check in_prop_lobec, not in_e
            prop_lobec = float(in_prop_lobec) if in_prop_lobec.strip() else DEFAULT_LOBEC

            in_f = input(f"Enter initial Fibroblasts [default {DEFAULT_F}]: ")
            initial_f = int(in_f) if in_f.strip() else DEFAULT_F

            in_m = input(f"Enter initial Macrophages [default {DEFAULT_M}]: ")
            initial_m = int(in_m) if in_m.strip() else DEFAULT_M
        except ValueError:
            print(f"Invalid input detected. Reverting to defaults ({DEFAULT_E}, {DEFAULT_LOBEC}, {DEFAULT_F}, {DEFAULT_M}).")
            initial_e, prop_lobec, initial_f, initial_m = DEFAULT_E, DEFAULT_LOBEC, DEFAULT_F, DEFAULT_M

    # --- Run ---
    history_data = run_simulation(initial_e, prop_lobec, initial_f, initial_m)

    # --- Export ---
    print("Exporting data to JSON...")

    output_data = {
        "config": {
            "FPS": FPS,
            "DT": DT,
            "SIM_DURATION": SIM_DURATION,
            "SCHEDULE": SCHEDULE,
            "SH_COLORS": SH_COLORS,
            "COLORS": COLORS,
            "E_HEIGHT": E_HEIGHT,
            "ECM_HEIGHT": ECM_HEIGHT,
            "F_HEIGHT": F_HEIGHT,
            "M_HEIGHT": M_HEIGHT,
            "INITIAL_COUNTS": {"E": initial_e, "F": initial_f, "M": initial_m}
        },
        "history": history_data
    }

    filename = f"simulation_data_E{initial_e}_F{initial_f}_M{initial_m}.json"
    with open(filename, "w") as f:
        json.dump(output_data, f)

    print(f"\tData saved to {filename}")

    # --- Instructions for Visualization ---
    print("\n" + "="*50)
    print("DONE. To visualize this result, run:")
    print(f"python3 visualization.py {filename}")
    print("="*50 + "\n")
