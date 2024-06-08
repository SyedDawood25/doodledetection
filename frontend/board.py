import pygame as pg
import random

pg.init()

class Board:

    def __init__(self, dim, pos=[0, 0]):
        self.rows = self.cols = dim
        self.pos = pos
        self.data = self._initData()

    def clearBoard(self):
        self.data = self._initData()

    def _initData(self):
        cols = []
        for x in range(0, self.rows, 20):
            row = []
            for y in range(0, self.cols, 20):
                row.append([pg.Rect(x, y, 20, 20), 0])
            cols.append(row)
        return cols
    
    def drawBoard(self, surface):
        self.boardSurface = pg.Surface((self.rows, self.cols))
        self._initBoard(self.boardSurface)
        self.boardBounds = surface.blit(self.boardSurface, (self.pos[0], self.pos[1]))

    def getData(self):
        data = []
        for row in self.data:
            for pixel in row:
                data.append(pixel[1])
        return data

    def _initBoard(self, surface):
        for row in self.data:
            for pixel in row:
                pg.draw.rect(surface, (pixel[1], pixel[1], pixel[1]), pixel[0])
    
    def draw(self, x, y):
        if not self.boardBounds.collidepoint(x, y):
            return

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j][0].collidepoint(x-self.boardBounds.x, y-self.boardBounds.y):
                    self.data[i][j][1] = 255
                    color = random.randrange(100, 200)
                    if i > 0 and i < len(self.data) - 1:
                        self.data[i-1][j][1] = color if self.data[i-1][j][1] == 0 else self.data[i-1][j][1]
                        self.data[i+1][j][1] = color if self.data[i+1][j][1] == 0 else self.data[i+1][j][1]
                    self._initBoard(self.boardSurface)