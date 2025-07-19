import random

def generate_reroute(current_pos):
    dx = random.randint(-100, 100)
    dy = random.randint(-100, 100)
    return (current_pos[0] + dx, current_pos[1] + dy)
