import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from stl import mesh
import numpy as np
import random

offsety_x = [23, 0, -23, -46] 
bazy_x = [32.6, 21.8]
wspolrzedne_z = [89.4, 83.4, 77.4, 71.4, 66.1]
wspolrzedne_y = [0, 4.345, 8.69, 13.035]

kolory_kontenerow = [(1.0, 0.1, 0.1), (0.1, 0.3, 0.8), (0.1, 0.6, 0.2), (1.0, 0.8, 0.0),(1.0, 0.5, 0.0), (0.5, 0.2, 0.7), (0.2, 0.2, 0.2), (0.9, 0.9, 0.9)]
kolory=[]
stala_falowania = 0.2

siatka_kontenerow = []
for oy in wspolrzedne_y:
    for ox in offsety_x:
        for bx in bazy_x:
            for wz in wspolrzedne_z:
                kolory.append(random.choice(kolory_kontenerow))
                siatka_kontenerow.append((bx - ox, wz, oy))

def wczytaj_stl(sciezka):
    moj_mesh = mesh.Mesh.from_file(sciezka)
    vertices = moj_mesh.vectors
    return vertices

def przygotuj_model(vertices):
    # Tworzy listę wyświetlania OpenGL w pamięci GPU
    lista_id = glGenLists(1)
    glNewList(lista_id, GL_COMPILE)
    glBegin(GL_TRIANGLES)
    for triangle in vertices:
        v1, v2, v3 = triangle[0], triangle[1], triangle[2]
        #Obliczanie wektora normalnego dla oswietlenia 3d
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        if norm > 0:
            glNormal3fv(normal / norm)
        for vertex in triangle:
            glVertex3fv(vertex)
    glEnd()
    glEndList()
    return lista_id

def rysuj_model(lista_id, x, y, z, r, g, b, skala=1.0, rot_y=0, rot_x=0, rot_z=0):
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    glRotatef(rot_z, 0, 0, 1)
    glScalef(skala, skala, skala)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [r, g, b, 1.0])
    glColor3f(r, g, b)
    glCallList(lista_id)
    glPopMatrix()

#model sześcianu
id_szescianu = None

def inicjalizuj_szescian():
    global id_szescianu
    id_szescianu = glGenLists(1)
    glNewList(id_szescianu, GL_COMPILE)
    v = [
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
    ]
    f = [(0,1,2,3), (4,5,6,7), (0,4,5,1), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    glBegin(GL_QUADS)
    for face in f:
        if face == (0,1,2,3): 
            glNormal3f(0,0,-1)
        elif face == (4,5,6,7): 
            glNormal3f(0,0,1)
        elif face == (0,4,5,1): 
            glNormal3f(0,-1,0)
        elif face == (2,3,7,6): 
            glNormal3f(0,1,0)
        elif face == (0,3,7,4): 
            glNormal3f(-1,0,0)
        elif face == (1,2,6,5): 
            glNormal3f(1,0,0)
        for i in face:
            glVertex3fv(v[i])
    glEnd()
    glEndList()
#inicjacja prostopadloscianu
def rysuj_prostopadloscian(x, y, z, szerokosc, wysokosc, glebokosc, r, g, b, rot_x=0, rot_y=0, rot_z=0):
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [r, g, b, 1.0])
    glColor3f(r, g, b)
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    glRotatef(rot_z, 0, 0, 1)
    glTranslatef(0, wysokosc / 2.0, 0)
    glScalef(szerokosc / 2.0, wysokosc / 2.0, glebokosc / 2.0)
    
    if id_szescianu is not None:
        glCallList(id_szescianu)
        
    glPopMatrix()
#model wody
#gestosc to jak bardzo ma byc szczegolowy model wody
#szerkosc i dlugosc to osie x i y (wyamiry)
class Woda:
    def __init__(self, szerokosc, dlugosc, gestosc):
        self.gestosc = gestosc
        self.count = gestosc * gestosc * 4
        self.skok = szerokosc / gestosc
        self.vertices = np.zeros((gestosc, gestosc, 4, 3), dtype=np.float32)
        for i in range(gestosc):
            for j in range(gestosc):
                x = -szerokosc/2 + i * self.skok
                z = -dlugosc/2 + j * self.skok
                self.vertices[i,j,0] = [x, 0, z]
                self.vertices[i,j,1] = [x + self.skok, 0, z]
                self.vertices[i,j,2] = [x + self.skok, 0, z + self.skok]
                self.vertices[i,j,3] = [x, 0, z + self.skok]
        self.vbo = glGenBuffers(1)

    def rysuj(self, czas, r, g, b):
        x_vals = self.vertices[:,:,:,0]
        z_vals = self.vertices[:,:,:,2]
        self.vertices[:,:,:,1] = np.sin(x_vals * 0.15 + czas) * stala_falowania + np.cos(z_vals * 0.2 + czas * 0.8) * 0.2
        #zabawa światłem
        glDisable(GL_LIGHTING)
        glColor4f(r, g, b, 0.9)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STREAM_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_QUADS, 0, self.count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glEnable(GL_LIGHTING)

def rysuj_niebo_i_slonce():
    glDisable(GL_LIGHTING)
    rysuj_prostopadloscian(200, 300, -400, 40, 40, 40, 1.0, 0.9, 0.0)
    glEnable(GL_LIGHTING)

def main():
    suwnica_x = 0.0
    pociag_x = 0.0
    wyciag_z = 0.0
    zejscie_y = 0        

    pygame.init()
    display = (1920, 1080)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    inicjalizuj_szescian()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, (display[0]/display[1]), 0.1, 1500.0)
    
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    
    bg_color = [0.1, 0.1, 0.15, 1]
    glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])

    glEnable(GL_LIGHTING)   
    glEnable(GL_LIGHT0)     
    glEnable(GL_COLOR_MATERIAL) 
    
    glLightfv(GL_LIGHT0, GL_POSITION,  [50, 100, 50, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT,   [0.2, 0.2, 0.2, 1]) 
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   [1.0, 1.0, 1.0, 1]) 
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, bg_color)
    glFogf(GL_FOG_DENSITY, 0.002)
    glHint(GL_FOG_HINT, GL_NICEST)

    cam_x, cam_y, cam_z = 7.5, 20, 120 
    rot_x, rot_y = 20, 0
    clock = pygame.time.Clock()
    t = 0.0

    niebo_kolor = [0.2, 0.4, 0.8, 1] 
    glClearColor(niebo_kolor[0], niebo_kolor[1], niebo_kolor[2], niebo_kolor[3])
    glLightfv(GL_LIGHT0, GL_POSITION, [200, 300, -400, 1])
    glFogfv(GL_FOG_COLOR, niebo_kolor)

    woda_obj = Woda(1400, 1200, 200)

    id_lodzi = None
    try:
        dane = wczytaj_stl('./models/CargoShip.stl')
        id_lodzi = przygotuj_model(dane)
    except:
        print("Nie znaleziono pliku CargoShip.stl!")

    id_kontenera = None
    try:
        dane = wczytaj_stl('./models/Container.stl')
        id_kontenera = przygotuj_model(dane)
    except:
        print("Nie znaleziono pliku Container.stl!")

    while True:
        t += 0.05
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                pygame.quit()
                return
        #ustawienie chywtaka
        chwytak_x = 7.5 + suwnica_x
        chwytak_z = 28.0 + wyciag_z
        chwytak_y = 46.0 - zejscie_y
        
        bujanie = math.sin(t * 0.5) * stala_falowania

        dx, dy = pygame.mouse.get_rel()
        rot_y += dx * 0.15
        rot_x = max(-89, min(89, rot_x + dy * 0.15))

        keys = pygame.key.get_pressed()
        rad = math.radians(rot_y)
        spd = 0.8 
        #sterowanie kamerą
        if keys[K_w] or keys[K_UP]:
            cam_x += math.sin(rad) * spd
            cam_z -= math.cos(rad) * spd
        if keys[K_s] or keys[K_DOWN]:
            cam_x -= math.sin(rad) * spd
            cam_z += math.cos(rad) * spd
        if keys[K_a]:
            cam_x -= math.cos(rad) * spd
            cam_z -= math.sin(rad) * spd
        if keys[K_d]:
            cam_x += math.cos(rad) * spd
            cam_z += math.sin(rad) * spd
        if keys[K_SPACE]:  cam_y += spd
        if keys[K_LSHIFT]: cam_y -= spd

        if keys[K_u]:
            for i, (k_x, k_z, k_y) in enumerate(siatka_kontenerow):
                kolor = kolory[i] if i < len(kolory) else (0, 0, 0)
                print(f"Kontener {i}: Pozycja X={k_x:.2f}, Z={k_z:.2f}, Y={k_y:.2f} | Kolor: {kolor}")

        if keys[K_n]: pociag_x += 0.1
        if keys[K_m]: pociag_x -= 0.1
        
        # Suwnica w lewo
        if keys[K_LEFT]:
            legalny_ruch = True
            przyszle_chwytak_x = chwytak_x + 0.05  
            for kx, kz, ky in siatka_kontenerow:
                realne_kontener_y = 8.2 + bujanie + ky
                if (abs(kx - przyszle_chwytak_x) < 7.0 and 
                    abs(kz - chwytak_z) < 6.2 and 
                    abs(realne_kontener_y - (chwytak_y + 1.5)) < 3.7):
                    legalny_ruch = False
                    break
            if suwnica_x < 70 and legalny_ruch:
                suwnica_x += 0.05

        # Suwnica w prawo
        if keys[K_RIGHT]:
            legalny_ruch = True
            przyszle_chwytak_x = chwytak_x - 0.05 
            for kx, kz, ky in siatka_kontenerow:
                realne_kontener_y = 8.2 + bujanie + ky
                if (abs(kx - przyszle_chwytak_x) < 7.0 and 
                    abs(kz - chwytak_z) < 6.2 and 
                    abs(realne_kontener_y - (chwytak_y + 1.5)) < 3.7):
                    legalny_ruch = False
                    break
            if suwnica_x > -20 and legalny_ruch:
                suwnica_x -= 0.05
                
        # Wózek w przód (Z)
        if keys[K_z]:
            legalny_ruch = True
            przyszle_chwytak_z = chwytak_z + 0.1  
            for kx, kz, ky in siatka_kontenerow:
                realne_kontener_y = 8.2 + bujanie + ky
                if (abs(kx - chwytak_x) < 14.0 and 
                    abs(kz - przyszle_chwytak_z) < 6.9 and 
                    abs(realne_kontener_y - (chwytak_y - 1.5)) < 4.345):
                    legalny_ruch = False
                    break
            if wyciag_z < 60 and legalny_ruch:
                wyciag_z += 0.1

        # Wózek w tył (X)
        if keys[K_x]:
            legalny_ruch = True
            przyszle_chwytak_z = chwytak_z - 0.1
            for kx, kz, ky in siatka_kontenerow:
                realne_kontener_y = 8.2 + bujanie + ky
                if (abs(kx - chwytak_x) < 7.0 and 
                    abs(kz - przyszle_chwytak_z) < 6.2 and 
                    abs(realne_kontener_y - (chwytak_y - 1.5)) < 3.7):
                    legalny_ruch = False
                    break
            if wyciag_z > -20 and legalny_ruch:
                wyciag_z -= 0.1
                
        # Opuszczanie chwytaka (V)
        if keys[K_v]:
            legalny_ruch = True
            przyszle_chwytak_y = chwytak_y - 0.1  
            for kx, kz, ky in siatka_kontenerow:
                realne_kontener_y = 8.2 + bujanie + ky
                if (abs(kx - chwytak_x) < 7.0 and 
                    abs(kz - chwytak_z) < 6.2 and 
                    abs(realne_kontener_y - (przyszle_chwytak_y - 1)) < 3.7):
                    legalny_ruch = False
                    break
            
            if zejscie_y < 50 and legalny_ruch:
                zejscie_y += 0.1
           
        # Podnoszenie chwytaka (C)
        if keys[K_c]:
            if zejscie_y > 0:
                zejscie_y -= 0.1

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        glTranslatef(-cam_x, -cam_y, -cam_z)

        # Pociag
        glPushMatrix() 
        glTranslatef(pociag_x, 0, 0)
        rysuj_prostopadloscian(23.375, 0.5, 12.5, 15, 5, 5, 0.9, 0.1, 0.1)
        rysuj_prostopadloscian(7.5, 0.5, 12.5, 15, 1, 5, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15.5, 0.75, 12.5, 1, 0.5, 0.5, 0.1, 0.1, 0.1)
        glPopMatrix()


        # Konstrukcja suwnicy
        glPushMatrix() 
        glTranslatef(suwnica_x, 0, 0)
        
        glPushMatrix() 
        glTranslatef(0, 0, wyciag_z)
        
        # Wyciag i lina
        rysuj_prostopadloscian(7.5, 48, 28, 10, 3, 7.5, 0.6, 0.6, 0.6)
        for i in range(1, int(zejscie_y*10)):
            if i % 10 == 0:
                rysuj_prostopadloscian(7.5, 48 - (i/10), 28, 0.3, 0.5, 0.3, 0, 0.9, 0.9)
        
        glPushMatrix() 
        glTranslatef(0, -zejscie_y, 0)
        
        #wykrywanie kolizji chwytaka z kontenerem
        mozna_podniesc = False
        y_renderu = 46
        
        for kx, kz, ky in siatka_kontenerow:
            realne_kontener_y = 8.2 + bujanie + ky
            if (chwytak_y - 5 < realne_kontener_y):
                if (abs(kx - chwytak_x) < 8 and abs(kz - chwytak_z) < 8):
                    mozna_podniesc = True
                    y_renderu = 46 + bujanie
                    #print("aaa")
                    break
                
        if mozna_podniesc:
            # zolty kolor gdy blisko
            rysuj_prostopadloscian(7.5, y_renderu, 28, 10, 3, 5.5, 1.0, 1.0, 0.0)
        else:
            # rozowy gdy daleko
            rysuj_prostopadloscian(7.5, 46, 28, 10, 3, 5.5, 0.9, 0.2, 0.6)
            
        glPopMatrix()
        glPopMatrix()

        # suwnica
        rysuj_prostopadloscian(0, 0, 0, 1, 52, 1, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 0, 0, 1, 52, 1, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(0, 0, 25, 1, 52, 1, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 0, 25, 1, 52, 1, 0.0, 0.0, 0.5)
        
        rysuj_prostopadloscian(0, 10, 12.5, 1, 3, 26, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 10, 12.5, 1, 3, 26, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(0, 25, 12.5, 0.5, 0.5, 26, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 25, 12.5, 0.5, 0.5, 26, 0.0, 0.0, 0.5)

        rysuj_prostopadloscian(0, 19, 6, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=45)
        rysuj_prostopadloscian(15, 19, 6, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=45)
        rysuj_prostopadloscian(0, 19, 19, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=-45)
        rysuj_prostopadloscian(15, 19, 19, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=-45)

        rysuj_prostopadloscian(0, 37.5, 12, 0.5, 0.8, 35, 0.0, 0.0, 0.5, rot_x=45)
        rysuj_prostopadloscian(15, 37.5, 12, 0.5, 0.8, 35, 0.0, 0.0, 0.5, rot_x=45)

        rysuj_prostopadloscian(7.5, 0, 0, 16, 1, 1, 0.8, 0.3, 0.6)
        rysuj_prostopadloscian(7.5, 0, 25, 16, 1, 1, 0.8, 0.3, 0.6)

        for z_pos in range(30, 100, 20):
            rysuj_prostopadloscian(2, 49, z_pos, 1, 3, 10, 0.9, 0, 0)
            rysuj_prostopadloscian(13, 49, z_pos, 1, 3, 10, 0.9, 0, 0)
            if z_pos + 10 <= 90:
                rysuj_prostopadloscian(2, 49, z_pos + 10, 1, 3, 10, 1, 1, 1)
                rysuj_prostopadloscian(13, 49, z_pos + 10, 1, 3, 10, 1, 1, 1)

        if math.floor(t) % 4 == 0:
            kolor_swiatla = (1, 1, 1) 
        else:
            kolor_swiatla = (0, 0, 0)  
            
        rysuj_prostopadloscian(2, 51.8, 94.8, 0.4, 0.4, 0.4, *kolor_swiatla)
        rysuj_prostopadloscian(13, 51.8, 94.8, 0.4, 0.4, 0.4, *kolor_swiatla)
        rysuj_prostopadloscian(7.5, 80, 25, 2, 2, 2, *kolor_swiatla)

        rysuj_prostopadloscian(2, 49, 12.5, 1.01, 3.01, 25, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(13, 49, 12.5, 1.01, 3.01, 25, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(7.5, 49, -30, 12, 3, 1, 0.0, 0.0, 0.5)

        rysuj_prostopadloscian(7.5, 49, 25, 16, 3, 1, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(7.5, 52, 0.5, 16, 3, 2,  0.0, 0.0, 0.5)
        
        rysuj_prostopadloscian(2, 49, -15, 1.01, 3.01, 30, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(13, 49, -15, 1.01, 3.01, 30, 0.0, 0.0, 0.5)

        rysuj_prostopadloscian(4.25, 66, 25, 29.5, 1, 1, 0.0, 0.0, 0.5, rot_z=75.5)
        rysuj_prostopadloscian(10.75 , 66, 25, 29.5, 1, 1, 0.0, 0.0, 0.5, rot_z=-75.5)
        rysuj_prostopadloscian(7.5, 67, 13, 1.5, 1.5, 35.5, 0.0, 0.0, 0.5, rot_x=-46)
        rysuj_prostopadloscian(7.5, 66, -2, 0.7, 0.7, 62, 0.0, 0.0, 0.5, rot_x=-28)

        rysuj_prostopadloscian(10.5, 66, 38, 0.3, 0.3, 39.5, 1, 0.0, 0, rot_x=47, rot_y=8)
        rysuj_prostopadloscian(4.5, 66, 38, 0.3, 0.3, 39.5, 1, 0.0, 0, rot_x=47, rot_y=-8)
        rysuj_prostopadloscian(10.5, 66, 55, 0.3, 0.3, 66, 1, 0.0, 0, rot_x=25.75, rot_y=5)
        rysuj_prostopadloscian(4.5, 66, 55, 0.3, 0.3, 66, 1, 0.0, 0, rot_x=25.75, rot_y=-5)

        rysuj_prostopadloscian(7.5, 50, -8, 10, 10, 15, 1, 1, 1)
        glPopMatrix()

        # Otoczenie port i woda
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        rysuj_prostopadloscian(0, -30, -45, 200, 30, 150, 0.8, 0.8, 0.8)
        rysuj_prostopadloscian(0, 0, 28, 200, 2, 4, 0.6, 0.6, 0.6)

        rysuj_prostopadloscian(30, 0, 0, 120, 0.1, 1, 0.3, 0, 0)
        rysuj_prostopadloscian(30, 0, 25, 120, 0.1, 1, 0.3, 0, 0)
        
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        lampka=-90
        for i in range(0, 20):
            rysuj_prostopadloscian(lampka, 2, 28, 0.7, 0.7, 0.7, 0.7, 0.4, 0)
            lampka+=10

        glPushMatrix()
        glTranslatef(7.5, -6, 600)
        woda_obj.rysuj(t, 0.0, 0.3, 0.7)
        glPopMatrix()

        glPushMatrix()
        glLoadIdentity()
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        rysuj_niebo_i_slonce()
        glPopMatrix()

        #statek i kontenery
        if id_lodzi is not None:
            bujanie = math.sin(t * 0.5) * stala_falowania 
            rysuj_model(id_lodzi, 30, -5 + bujanie, 80, 0.5, 0.3, 0.1, skala=0.8, rot_x=-90, rot_z=90)
            i=0
            if id_kontenera is not None:
                for k_x, k_z, k_y in siatka_kontenerow:
                    rysuj_model(
                        id_kontenera, 
                        k_x, 8.2 + bujanie + k_y, k_z, 
                        kolory[i][0], kolory[i][1], kolory[i][2],
                        skala=0.24, 
                        rot_x=-90, 
                        rot_z=90
                    )
                    i+=1
                
        pygame.display.flip()
        clock.tick(120)
if __name__ == "__main__":
    main()