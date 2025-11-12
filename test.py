import random as rand
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

def circle_points(total_points, origins, radius, instances, num_circles):
    os.makedirs(f'Circle-points/{num_circles}-circle/{total_points}-points/{radius}-radius', exist_ok=True)
    rand.seed("None")
    angles = []
    points = []

    for j in range(instances):
        for origin in origins:
            # generate random angles
            while len(angles) < total_points:
                angles.append(rand.random() * 2 * math.pi)

            for ang in angles:
                x = radius * math.cos(ang) + origin[0]
                y = radius * math.sin(ang) + origin[1]
                points.append((x,y))

            angles.clear()

        data = pd.DataFrame(points, columns=['x','y'])
        data.to_csv(f'Circle-points/{num_circles}-circle/{total_points}-points/{radius}-radius/instance{j}', float_format='%.7e', sep=' ', index=False, header=None)

        points.clear()



def main():
    circle_points(128, [[0,0]], 1, 8, 1)
    circle_points(128, [[0,0], [8,0]], 1, 7, 2)

if __name__ == '__main__':
  main()