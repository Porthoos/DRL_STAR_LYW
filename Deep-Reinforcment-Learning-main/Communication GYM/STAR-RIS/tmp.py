#TODO userposition is 6*3, RISPosition is 1*3, BSPosition is 1*3
#TODO using YB function / LYW function? convex optimization
#TODO return 1*3, within 1m circle?
import copy

import numpy.random
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '2'


# users random move within 0.5m
# look for present position, make a move for RIS
# look for 3 steps position, using random to predict, make a move for RIS
# run total like 3/50 steps, observe total reward.
# for 3 steps prediction, totally 6^(8*8*8), read all possible positions, choose the best position

#TODO generate best position
def getLoc(userPosition, RISPosition, BSPosition):
    position = [np.mean(userPosition[0]), np.mean(userPosition[1])]
    return position


def getReward(userPosition, RISPosition, BSPosition):
    # print(userPosition)
    # print(RISPosition)
    # print(BSPosition)
    reward = 6*((BSPosition[0]-RISPosition[0])**2+(BSPosition[1]-RISPosition[1])**2)**0.5
    reward += np.sum(((userPosition[0]-RISPosition[0])**2+(userPosition[1]-RISPosition[1])**2)**0.5)
    return reward


#TODO make random move for users
def randomMove(userPosition):
    nextPosition = userPosition + numpy.random.normal(scale=3, size=(2,6))
    return nextPosition


def evaluateOneStep(BSPosition, userPosition, RISPosition):
    totalReward = 0
    for i in range(50):

        loc = getLoc(userPosition, RISPosition, BSPosition)

        userPosition = randomMove(userPosition)
        totalReward += getReward(userPosition, RISPosition, BSPosition)
        RISPosition = loc

    return totalReward

def evaluateThreeSteps(BSPosition, userPosition, RISPosition):
    totalReward = 0
    for i in range(50):

        loc = [0,0]
        # for j in range(100):
        #     tmpPosition = randomMove(randomMove(randomMove(userPosition)))
        #     loc += np.array(getLoc(tmpPosition, RISPosition, BSPosition))/np.array([100,100])
        a, b, c = 0.7, 0.2, 0.1
        for j in range(100):
            tmpPosition = randomMove(userPosition)
            # loc.append(getLoc(tmpPosition, RISPosition, BSPosition))
            loc += np.array(getLoc(tmpPosition, RISPosition, BSPosition))/np.array([100/a,100/a])
            tmpPosition = randomMove(tmpPosition)
            # loc.append(getLoc(tmpPosition, RISPosition, BSPosition))
            loc += np.array(getLoc(tmpPosition, RISPosition, BSPosition))/np.array([100/b,100/b])
            tmpPosition = randomMove(tmpPosition)
            # loc.append(getLoc(tmpPosition, RISPosition, BSPosition))
            loc += np.array(getLoc(tmpPosition, RISPosition, BSPosition))/np.array([100/c,100/c])

        userPosition = randomMove(userPosition)
        totalReward += getReward(userPosition, RISPosition, BSPosition)
        # model.fit(loc)
        # RISPosition = model.cluster_centers_[0]
        RISPosition = loc

    return totalReward



totalReward = 0
BSPosition = [0,0]
#TODO initial user position, users inial at random position in the circle of R=10, center of RIS
userPosition = [[99, 98, 97, 101, 102, 103],
                [99, 98, 97, 101, 102, 103]]
#TODO initial RIS position, both method start at same position, thinking as center of a circle
RISPosition = [100,100]

countA = 0
countB = 0
for p in range(10):
    countA, countB = 0, 0
    for i in range(100):
        RA = evaluateOneStep(BSPosition, userPosition, RISPosition)
        RB = evaluateThreeSteps(BSPosition, userPosition, RISPosition)
        # print(RA, RB)
        countA += (RA>RB)
        countB += (RB>RA)
    print(countA, countB)