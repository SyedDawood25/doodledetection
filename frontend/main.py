import pygame as pg
from board import Board
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import NeuralNetworkCPU

nn = None
with open('C:\\Users\\Syed Dawood\\Documents\\MLProject\\frontend\\trained_nn\\trained_nn_final.data', 'rb') as f:
    nn = pickle.loads(f.read())

pg.init()

# Initialize
WIDTH, HEIGHT = 1000, 800
display = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption('Doodle Detection')
clock = pg.time.Clock()

WHITE = (255, 255, 255)
running = True
mouseDown = False
predictedClass = 'Start Drawing :)'
classes = ['Airplane', 'Axe', 'Basket Ball', 'Bicycle', 'Camera', 'Diamond', 'Donut', 'Envelope', 'Fish', 'Helicopter']
confidencePerClass = ['Airplane - 0%', 'Axe - 0%', 'Basket Ball - 0%', 'Bicycle - 0%', 'Camera - 0%', 'Diamond - 0%', 'Donut - 0%', 'Envelope - 0%', 'Fish - 0%', 'Helicopter - 0%']

# Objects
gameBoard = Board(560, (50, 200))

def createUI():
    font = pg.font.Font('C:\\Users\\Syed Dawood\\Documents\\MLProject\\frontend\\Astonpoliz.ttf', size=50)
    text = font.render('A Neural Network', True, (200, 0, 0))
    text2 = font.render('To Detect Doodles', True, (0, 0, 0))
    textRect = text2.get_rect()
    display.blit(text, ((WIDTH // 2) - (textRect.width // 2), 40))
    display.blit(text2, ((WIDTH // 2) - (textRect.width // 2), 90))
    pg.draw.rect(display, (255, 255, 255), (630, 200, 350, 50), border_radius=250)
    pg.draw.rect(display, (0, 0, 0), (630, 200, 350, 50), width=2, border_radius=250)
    pg.draw.rect(display, (255, 255, 255), (630, 270, 350, 490), border_radius=20)
    pg.draw.rect(display, (0, 0, 0), (630, 270, 350, 490), width=2, border_radius=20)
    font2 = pg.font.Font('C:\\Users\\Syed Dawood\\Documents\\MLProject\\frontend\\Astonpoliz.ttf', size=28)
    predictedClassText = font2.render(predictedClass, True, (0, 0, 0))
    textRect = predictedClassText.get_rect()
    display.blit(predictedClassText, (805 - (textRect.width // 2), 210))
    y = 320
    for confidence in confidencePerClass:
        color = (0, 200, 0) if confidence.__contains__(predictedClass) else (200, 0, 0)
        text = font2.render(confidence, True, color)
        textRect = text.get_rect()
        display.blit(text, (805 - (textRect.width // 2), y))
        y += textRect.height + 10

def prepareConfidenceList(confidences):
    for i in range(len(confidences)):
        confidencePerClass[i] = classes[i] + " - " + str(round(confidences[i] * 100, 3)) + "%"

while running:

    display.fill(WHITE)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
            break
        elif event.type == pg.MOUSEBUTTONDOWN:
            mouseDown = True
        elif event.type == pg.MOUSEBUTTONUP:
            mouseDown = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_c:
                gameBoard.clearBoard()
    
    if mouseDown:
        x, y = pg.mouse.get_pos()
        gameBoard.draw(x, y)
        data = np.array(gameBoard.getData()).reshape((28, 28)).T.flatten()
        data = StandardScaler().fit_transform(data.reshape((-1, 1))).flatten()
        confidences, prediction = nn.predict(data)
        predictedClass = classes[prediction[0]]
        prepareConfidenceList(confidences[0])
    
    gameBoard.drawBoard(display)
    createUI()

    pg.display.flip()
    clock.tick(60)

pg.quit()