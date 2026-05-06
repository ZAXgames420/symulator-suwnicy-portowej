import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from stl import mesh
import numpy as np
import random



offsety_x = [23, 0, -23, -46] 

# Bazowe współrzędne X
bazy_x = [32.6, 21.8]

# Wszystkie współrzędne Z z Twojego kodu
wspolrzedne_z = [89.4, 83.4, 77.4, 71.4, 66.1]

wspolrzedne_y = [0, 4.345, 8.69, 13.035]

kolory_kontenerow = [(1.0, 0.1, 0.1), (0.1, 0.3, 0.8), (0.1, 0.6, 0.2), (1.0, 0.8, 0.0),(1.0, 0.5, 0.0), (0.5, 0.2, 0.7), (0.2, 0.2, 0.2), (0.9, 0.9, 0.9)]

kolory=[]

# Generujemy listę wszystkich par (x, z, y)
siatka_kontenerow = []
for oy in wspolrzedne_y:
    for ox in offsety_x:
        for bx in bazy_x:
            for wz in wspolrzedne_z:
                kolory.append(random.choice(kolory_kontenerow))
                siatka_kontenerow.append((bx - ox, wz, oy))

def wczytaj_stl(sciezka):
    # Wczytujemy dane z pliku .stl
    moj_mesh = mesh.Mesh.from_file(sciezka)
    # Wyciągamy punkty (v0, v1, v2) dla każdego trójkąta
    # numpy-stl przechowuje je w formacie [liczba_trojkatow, 9]
    vertices = moj_mesh.vectors
    return vertices

def przygotuj_model_gpu(vertices):
    """Tworzy Display List dla łodzi, aby obciążyć GPU zamiast CPU."""
    lista_id = glGenLists(1)
    glNewList(lista_id, GL_COMPILE)
    glBegin(GL_TRIANGLES)
    for triangle in vertices:
        v1, v2, v3 = triangle[0], triangle[1], triangle[2]
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        if norm > 0:
            glNormal3fv(normal / norm)
        for vertex in triangle:
            glVertex3fv(vertex)
    glEnd()
    glEndList()
    return lista_id

def rysuj_model_gpu(lista_id, x, y, z, r, g, b, skala=1.0, rot_y=0, rot_x=0, rot_z=0):
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


def rysuj_obrocony_prostopadloscian(x, y, z, szerokosc, wysokosc, glebokosc, r, g, b, rot_x=0, rot_y=0, rot_z=0):
    """
    Rysuje prostopadłościan z rotacją. (x, y, z) to środek dolnej podstawy.
    """
    # Ustawienie koloru materiału, aby reagował na światło
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [r, g, b, 1.0])
    glColor3f(r, g, b)
    glPushMatrix()
    
    # 1. Przesunięcie do pozycji docelowej w świecie
    glTranslatef(x, y, z)
    
    # 2. Obroty (wykonywane względem punktu x, y, z)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    glRotatef(rot_z, 0, 0, 1)
    
    # 3. Przesunięcie modelu o połowę wysokości w górę, aby punkt obrotu był na dole
    glTranslatef(0, wysokosc / 2.0, 0)
    
    # 4. Skalowanie
    glScalef(szerokosc / 2.0, wysokosc / 2.0, glebokosc / 2.0)
    
    # Dane wierzchołków
    v = [
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
    ]
    f = [(0,1,2,3), (4,5,6,7), (0,4,5,1), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    
    glBegin(GL_QUADS)
    for face in f:
        # Dodanie wektorów normalnych dla każdej ściany (niezbędne do poprawnego cieniowania)
        if face == (0,1,2,3): glNormal3f(0,0,-1)
        elif face == (4,5,6,7): glNormal3f(0,0,1)
        elif face == (0,4,5,1): glNormal3f(0,-1,0)
        elif face == (2,3,7,6): glNormal3f(0,1,0)
        elif face == (0,3,7,4): glNormal3f(-1,0,0)
        elif face == (1,2,6,5): glNormal3f(1,0,0)
        for i in face:
            glVertex3fv(v[i])
    glEnd()
    glPopMatrix()

class WodaGPU:
    """Klasa obsługująca wodę za pomocą VBO dla maksymalnej wydajności."""
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
        self.vertices[:,:,:,1] = np.sin(x_vals * 0.15 + czas) * 0.5 + np.cos(z_vals * 0.2 + czas * 0.8) * 0.5
        
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
    # --- SŁOŃCE ---
    glDisable(GL_LIGHTING) # Słońce samo świeci
    # Rysujemy słońce bardzo wysoko i daleko
    rysuj_obrocony_prostopadloscian(200, 300, -400, 40, 40, 40, 1.0, 0.9, 0.0)
    
    # --- NIEBO ---
    # Ogromny sześcian otaczający świat (nie reaguje na światło)
    # Kolor góry nieba (jasnoniebieski)
    rysuj_obrocony_prostopadloscian(0, -500, 0, 2000, 2000, 2000, 0.2, 0.4, 0.8)
    glEnable(GL_LIGHTING)

def main():
    suwnica_x = 0.0
    pociag_x = 0.0
    wyciag_z = 0.0

    pygame.init()
    display = (1920, 1080)
    #muzyczka
    #file = 'C:/Users/Damciu/Desktop/muzyczka.mp3'
    #pygame.mixer.init()
    #pygame.mixer.music.load(file)
    #pygame.mixer.music.play()

    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, (display[0]/display[1]), 0.1, 1500.0)
    
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    
    # Kolor tła dopasowany do mgły
    bg_color = [0.1, 0.1, 0.15, 1]
    glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])

    # === KONFIGURACJA OŚWIETLENIA ===
    glEnable(GL_LIGHTING)   # Włącz oświetlenie
    glEnable(GL_LIGHT0)     # Włącz pierwsze źródło światła
    glEnable(GL_COLOR_MATERIAL) # Pozwól glColor kontrolować kolory przy oświetleniu
    
    # Parametry światła (pozycja: x, y, z, w) - w=0 oznacza światło kierunkowe (jak słońce)
    glLightfv(GL_LIGHT0, GL_POSITION,  [50, 100, 50, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT,   [0.2, 0.2, 0.2, 1]) # Światło otoczenia (cienie nie są czarne)
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   [1.0, 1.0, 1.0, 1]) # Światło główne (białe)
    
    # === KONFIGURACJA PRZEZROCZYSTOŚCI I MGŁY ===
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

    # Pasuje do koloru nieba
    niebo_kolor = [0.2, 0.4, 0.8, 1] 
    glClearColor(niebo_kolor[0], niebo_kolor[1], niebo_kolor[2], niebo_kolor[3])
    
    # Pozycja światła tam gdzie narysowane słońce [x, y, z, w]
    glLightfv(GL_LIGHT0, GL_POSITION, [200, 300, -400, 1])
    
    # Mgła też musi mieć kolor nieba!
    glFogfv(GL_FOG_COLOR, niebo_kolor)

    # Inicjalizacja wody na GPU
    woda_obj = WodaGPU(1400, 1200, 40)

    # Inicjalizacja łodzi na GPU
    id_lodzi = None
    try:
        dane = wczytaj_stl('models/CargoShip.stl')
        id_lodzi = przygotuj_model_gpu(dane)
    except:
        print("Nie znaleziono pliku CargoShip.stl!")

    # Inicjalizacja kontenera na GPU
    id_kontenera = None
    try:
        dane = wczytaj_stl('models/Container.stl')
        id_kontenera = przygotuj_model_gpu(dane)
    except:
        print("Nie znaleziono pliku Container.stl!")

    while True:
        t += 0.05
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                pygame.quit()
                return

        dx, dy = pygame.mouse.get_rel()
        rot_y += dx * 0.15
        rot_x = max(-89, min(89, rot_x + dy * 0.15))

        keys = pygame.key.get_pressed()
        rad = math.radians(rot_y)
        spd = 0.8 # Przyspieszyłem kamerę dla wygody
        
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
        #suwnica prawo lewo
        if keys[K_LEFT]:
            if(suwnica_x<70):
                suwnica_x += 0.1
        if keys[K_RIGHT]:
            if(suwnica_x>-20):
                suwnica_x -= 0.1
        #pociąg w przód i tył
        if keys[K_n]:
            pociag_x += 0.1
        if keys[K_m]:
            pociag_x -= 0.1
        #wyciąg w przód i tył
        if keys[K_z]:
            if(wyciag_z<60):
                wyciag_z += 0.1
        if keys[K_x]:
            if(wyciag_z>-20):
                wyciag_z -= 0.1



        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        glTranslatef(-cam_x, -cam_y, -cam_z)

        #Pociąg
        glPushMatrix() 
        glTranslatef(pociag_x, 0, 0)
        #ciufcia
        rysuj_obrocony_prostopadloscian(23.375, 0.5, 12.5, 15, 5, 5, 0.9, 0.1, 0.1)
        #wagony
        for nr in range(0, 10):
            przes=nr*15.875
            rysuj_obrocony_prostopadloscian(7.5 - przes, 0.5, 12.5, 15, 1, 5, 0.0, 0.0, 0.5)
            rysuj_obrocony_prostopadloscian(15.5 - przes, 0.75, 12.5, 1, 0.5, 0.5, 0.1, 0.1, 0.1)

        glPopMatrix()

        #tory
        rysuj_obrocony_prostopadloscian(0, 0.1, 14.4, 200, 0.2, 0.2, 0, 0, 0)
        rysuj_obrocony_prostopadloscian(0, 0.1, 10.6, 200, 0.2, 0.2, 0, 0, 0)

        # === SZKIELET KONSTRUKCJI DŹWIGU ===
        glPushMatrix() 
        glTranslatef(suwnica_x, 0, 0)

        glPushMatrix() 
        glTranslatef(0, 0, wyciag_z)
        # wyciąg
        zejscie = 10
        rysuj_obrocony_prostopadloscian(7.5, 48, 28, 10, 3, 7.5, 0.6, 0.6, 0.6)
        for i in range(1, zejscie):
            rysuj_obrocony_prostopadloscian(7.5, 48-i, 28, 0.3, 0.1, 0.3, 0, 0.9, 0.9)

        glPopMatrix()

        # Kolumny główne
        rysuj_obrocony_prostopadloscian(0, 0, 0, 1, 52, 1, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(15, 0, 0, 1, 52, 1, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(0, 0, 25, 1, 52, 1, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(15, 0, 25, 1, 52, 1, 0.0, 0.0, 0.5)
        
        #belki tir 1
        rysuj_obrocony_prostopadloscian(0, 10, 12.5, 1, 3, 26, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(15, 10, 12.5, 1, 3, 26, 0.0, 0.0, 0.5)

        #belki tir 2 male
        rysuj_obrocony_prostopadloscian(0, 25, 12.5, 0.5, 0.5, 26, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(15, 25, 12.5, 0.5, 0.5, 26, 0.0, 0.0, 0.5)

        #belki tir ukos 45
        rysuj_obrocony_prostopadloscian(0, 19, 6, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=45)
        rysuj_obrocony_prostopadloscian(15, 19, 6, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=45)
        rysuj_obrocony_prostopadloscian(0, 19, 19, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=-45)
        rysuj_obrocony_prostopadloscian(15, 19, 19, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=-45)

        #belki tir ukos duze
        rysuj_obrocony_prostopadloscian(0, 37.5, 12, 0.5, 0.8, 35, 0.0, 0.0, 0.5, rot_x=45)
        rysuj_obrocony_prostopadloscian(15, 37.5, 12, 0.5, 0.8, 35, 0.0, 0.0, 0.5, rot_x=45)

        # Podstawa dolna
        rysuj_obrocony_prostopadloscian(7.5, 0, 0, 16, 1, 1, 0.8, 0.3, 0.6)
        rysuj_obrocony_prostopadloscian(7.5, 0, 25, 16, 1, 1, 0.8, 0.3, 0.6)

        #szyny ramienia
        for z_pos in range(30, 100, 20):
            rysuj_obrocony_prostopadloscian(2, 49, z_pos, 1, 3, 10, 0.9, 0, 0)
            rysuj_obrocony_prostopadloscian(13, 49, z_pos, 1, 3, 10, 0.9, 0, 0)
            if z_pos + 10 <= 90:
                rysuj_obrocony_prostopadloscian(2, 49, z_pos + 10, 1, 3, 10, 1, 1, 1)
                rysuj_obrocony_prostopadloscian(13, 49, z_pos + 10, 1, 3, 10, 1, 1, 1)

        #światełka na czerwono-białych ramieniach
        # Migotanie światełek
        if math.floor(t) % 4 == 0:
            kolor_swiatla = (1, 1, 1) 
        else:
            kolor_swiatla = (0, 0, 0)  
        #lampki na czerwono-białych ramionach
        rysuj_obrocony_prostopadloscian(2, 51.8, 94.8, 0.4, 0.4, 0.4, *kolor_swiatla)
        rysuj_obrocony_prostopadloscian(13, 51.8, 94.8, 0.4, 0.4, 0.4, *kolor_swiatla)
        #lampka na czubku
        rysuj_obrocony_prostopadloscian(7.5, 80, 25, 2, 2, 2, *kolor_swiatla)

        #gorne belki w srodku
        rysuj_obrocony_prostopadloscian(2, 49, 12.5, 1.01, 3.01, 25, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(13, 49, 12.5, 1.01, 3.01, 25, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(7.5, 49, -30, 12, 3, 1, 0.0, 0.0, 0.5)

        #belki trzymające wyciąg
        rysuj_obrocony_prostopadloscian(7.5, 49, 25, 16, 3, 1, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(7.5, 52, 0.5, 16, 3, 2,  0.0, 0.0, 0.5)
        
        #gorne belki do odwaznika
        rysuj_obrocony_prostopadloscian(2, 49, -15, 1.01, 3.01, 30, 0.0, 0.0, 0.5)
        rysuj_obrocony_prostopadloscian(13, 49, -15, 1.01, 3.01, 30, 0.0, 0.0, 0.5)

        #gorne belki łączące
        rysuj_obrocony_prostopadloscian(4.25, 66, 25, 29.5, 1, 1, 0.0, 0.0, 0.5, rot_z=75.5)
        rysuj_obrocony_prostopadloscian(10.75 , 66, 25, 29.5, 1, 1, 0.0, 0.0, 0.5, rot_z=-75.5)
        rysuj_obrocony_prostopadloscian(7.5, 67, 13, 1.5, 1.5, 35.5, 0.0, 0.0, 0.5, rot_x=-46)
        rysuj_obrocony_prostopadloscian(7.5, 66, -2, 0.7, 0.7, 62, 0.0, 0.0, 0.5, rot_x=-28)

        #linki wyciągu
        rysuj_obrocony_prostopadloscian(10.5, 66, 38, 0.3, 0.3, 39.5, 1, 0.0, 0, rot_x=47, rot_y=8)
        rysuj_obrocony_prostopadloscian(4.5, 66, 38, 0.3, 0.3, 39.5, 1, 0.0, 0, rot_x=47, rot_y=-8)
        rysuj_obrocony_prostopadloscian(10.5, 66, 55, 0.3, 0.3, 66, 1, 0.0, 0, rot_x=25.75, rot_y=5)
        rysuj_obrocony_prostopadloscian(4.5, 66, 55, 0.3, 0.3, 66, 1, 0.0, 0, rot_x=25.75, rot_y=-5)

        #odważnik
        rysuj_obrocony_prostopadloscian(7.5, 50, -8, 10, 10, 15, 1, 1, 1)

        glPopMatrix()

        #=== LĄD ===
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        rysuj_obrocony_prostopadloscian(0, -30, -45, 200, 30, 150, 0.8, 0.8, 0.8)
        rysuj_obrocony_prostopadloscian(0, 0, 28, 200, 2, 4, 0.6, 0.6, 0.6)

        #szyny suwnicy
        rysuj_obrocony_prostopadloscian(30, 0, 0, 120, 0.1, 1, 0.3, 0, 0)
        rysuj_obrocony_prostopadloscian(30, 0, 25, 120, 0.1, 1, 0.3, 0, 0)
        
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        #lampki na brzegu
        lampka=-90
        for i in range(0, 20):
            rysuj_obrocony_prostopadloscian(lampka, 2, 28, 0.7, 0.7, 0.7, 0.7, 0.4, 0)
            lampka+=10

        #woda
        glPushMatrix()
        glTranslatef(7.5, -6, 600)
        woda_obj.rysuj(t, 0.0, 0.3, 0.7)
        glPopMatrix()

        #niebo i słońce
        glPushMatrix()
        glLoadIdentity()
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        rysuj_niebo_i_slonce()
        glPopMatrix()

        #łódź
        if id_lodzi is not None:
            bujanie = math.sin(t * 0.5) * 0.5
            rysuj_model_gpu(id_lodzi, 30, -5 + bujanie, 80, 0.5, 0.3, 0.1, skala=0.8, rot_x=-90, rot_z=90)
            i=0
            if id_kontenera is not None:
                for k_x, k_z, k_y in siatka_kontenerow:
                    
                    rysuj_model_gpu(
                        id_kontenera, 
                        k_x, 8.2 + bujanie + k_y, k_z, 
                        kolory[i][0], kolory[i][1], kolory[i][2], # kolor czerwony
                        skala=0.24, 
                        rot_x=-90, 
                        rot_z=90
                    )
                    i+=1
                


        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()