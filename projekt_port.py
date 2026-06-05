import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from stl import mesh
import numpy as np
import random

# ─────────────────────────────────────────────
# GLOBALNE STAŁE
# ─────────────────────────────────────────────
STALA_FALOWANIA   = 0.2
OFFSETY_X         = [23, 0, -23, -46]
BAZY_X            = [32.6, 21.8]
WSPOLRZEDNE_Z     = [89.4, 83.4, 77.4, 71.4, 66.1]
WSPOLRZEDNE_Y     = [0, 4.345, 8.69, 13.035]
KOLORY_KONTENEROW = [
    (1.0, 0.1, 0.1), (0.1, 0.3, 0.8), (0.1, 0.6, 0.2), (1.0, 0.8, 0.0),
    (1.0, 0.5, 0.0), (0.5, 0.2, 0.7), (0.2, 0.2, 0.2), (0.9, 0.9, 0.9),
]

# ─────────────────────────────────────────────
# POMOCNICZE FUNKCJE OPENGL  (niezmienione)
# ─────────────────────────────────────────────
_id_szescianu = None

def wczytaj_stl(sciezka):
    moj_mesh = mesh.Mesh.from_file(sciezka)
    return moj_mesh.vectors

def przygotuj_model(vertices):
    lista_id = glGenLists(1)
    glNewList(lista_id, GL_COMPILE)
    glBegin(GL_TRIANGLES)
    for triangle in vertices:
        v1, v2, v3 = triangle[0], triangle[1], triangle[2]
        normal = np.cross(v2 - v1, v3 - v1)
        norm   = np.linalg.norm(normal)
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

def inicjalizuj_szescian():
    global _id_szescianu
    _id_szescianu = glGenLists(1)
    glNewList(_id_szescianu, GL_COMPILE)
    v = [
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
        [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1],
    ]
    f = [(0,1,2,3),(4,5,6,7),(0,4,5,1),(2,3,7,6),(0,3,7,4),(1,2,6,5)]
    glBegin(GL_QUADS)
    for face in f:
        if   face == (0,1,2,3): glNormal3f( 0, 0,-1)
        elif face == (4,5,6,7): glNormal3f( 0, 0, 1)
        elif face == (0,4,5,1): glNormal3f( 0,-1, 0)
        elif face == (2,3,7,6): glNormal3f( 0, 1, 0)
        elif face == (0,3,7,4): glNormal3f(-1, 0, 0)
        elif face == (1,2,6,5): glNormal3f( 1, 0, 0)
        for i in face:
            glVertex3fv(v[i])
    glEnd()
    glEndList()

def rysuj_prostopadloscian(x, y, z, szerokosc, wysokosc, glebokosc,
                            r, g, b, rot_x=0, rot_y=0, rot_z=0):
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [r, g, b, 1.0])
    glColor3f(r, g, b)
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    glRotatef(rot_z, 0, 0, 1)
    glTranslatef(0, wysokosc / 2.0, 0)
    glScalef(szerokosc / 2.0, wysokosc / 2.0, glebokosc / 2.0)
    if _id_szescianu is not None:
        glCallList(_id_szescianu)
    glPopMatrix()

def rysuj_niebo_i_slonce():
    glDisable(GL_LIGHTING)
    rysuj_prostopadloscian(200, 300, -400, 40, 40, 40, 1.0, 0.9, 0.0)
    glEnable(GL_LIGHTING)

# ═════════════════════════════════════════════
# KLASY
# ═════════════════════════════════════════════

class Woda:
    """Animowana powierzchnia wody — niezmieniona."""
    def __init__(self, szerokosc, dlugosc, gestosc):
        self.gestosc = gestosc
        self.count   = gestosc * gestosc * 4
        self.skok    = szerokosc / gestosc
        self.vertices = np.zeros((gestosc, gestosc, 4, 3), dtype=np.float32)
        for i in range(gestosc):
            for j in range(gestosc):
                x = -szerokosc / 2 + i * self.skok
                z = -dlugosc   / 2 + j * self.skok
                self.vertices[i, j, 0] = [x,            0, z]
                self.vertices[i, j, 1] = [x + self.skok, 0, z]
                self.vertices[i, j, 2] = [x + self.skok, 0, z + self.skok]
                self.vertices[i, j, 3] = [x,            0, z + self.skok]
        self.vbo = glGenBuffers(1)

    def rysuj(self, czas, r, g, b):
        x_vals = self.vertices[:, :, :, 0]
        z_vals = self.vertices[:, :, :, 2]
        self.vertices[:, :, :, 1] = (
            np.sin(x_vals * 0.15 + czas) * STALA_FALOWANIA
            + np.cos(z_vals * 0.2 + czas * 0.8) * 0.2
        )
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


class Kamera:
    """Swobodna kamera FPS."""
    def __init__(self):
        self.x     = 7.5
        self.y     = 20.0
        self.z     = 120.0
        self.rot_x = 20.0
        self.rot_y = 0.0

    def obsluz_mysz(self, dx, dy):
        self.rot_y += dx * 0.15
        self.rot_x  = max(-89, min(89, self.rot_x + dy * 0.15))

    def obsluz_klawisze(self, keys):
        spd = 0.8
        rad = math.radians(self.rot_y)
        if keys[K_w] or keys[K_UP]:
            self.x += math.sin(rad) * spd
            self.z -= math.cos(rad) * spd
        if keys[K_s] or keys[K_DOWN]:
            self.x -= math.sin(rad) * spd
            self.z += math.cos(rad) * spd
        if keys[K_a]:
            self.x -= math.cos(rad) * spd
            self.z -= math.sin(rad) * spd
        if keys[K_d]:
            self.x += math.cos(rad) * spd
            self.z += math.sin(rad) * spd
        if keys[K_SPACE]:  self.y += spd
        if keys[K_LSHIFT]: self.y -= spd

    def zastosuj(self):
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glTranslatef(-self.x, -self.y, -self.z)


class Chwytak:
    """Chwytak — opuszczanie, podnoszenie, łapanie kontenera, zmiana koloru."""
    def __init__(self):
        self.zejscie_y = 0.0
        # pozycja w świecie (ustawiana przez Suwnicę każdą klatkę)
        self.x = 7.5
        self.y = 46.0
        self.z = 28.0
        # podnoszenie
        self.podniesiony_idx    = None
        self.podniesiony_offset = 0.0   # -(1.5 + 3.0)

    # ── warunek złapania (identyczny z wykrywaniem koloru) ──────────────
    def _warunek_zlapania(self, kx, kz, ky, bujanie):
        realne_y   = 8.2 + bujanie + ky
        gora       = realne_y + 3.0
        chwytak_dol = self.y - 1.5
        return (
            abs((kx - 3) - self.x) < 2.0 and
            abs((kz - 1) - self.z) < 2.0 and
            chwytak_dol > gora - 1.0 and
            chwytak_dol < gora + 1.0
        )

    def probuj_zlapac(self, siatka_kontenerow, bujanie):
        chwytak_dol = self.y - 1.5
        for idx, (kx, kz, ky) in enumerate(siatka_kontenerow):
            realne_y = 8.2 + bujanie + ky
            gora     = realne_y + 3.0
            if (abs((kx - 3) - self.x) < 4.0 and
                abs((kz - 1) - self.z) < 4.0 and
                chwytak_dol > gora - 1.0 and
                chwytak_dol < gora + 1.0):
                self.podniesiony_idx    = idx
                self.podniesiony_offset = -(1.5 + 3.0)
                return

    def upusc(self):
        self.podniesiony_idx    = None
        self.podniesiony_offset = 0.0

    def opusc(self, siatka_kontenerow, bujanie):
        """Klawisz V — opuszczanie z kolizją."""
        przyszle_y = self.y - 0.1
        for kx, kz, ky in siatka_kontenerow:
            realne_y = 8.2 + bujanie + ky
            if (abs(kx - self.x) < 7.0 and
                abs(kz - self.z) < 6.2 and
                abs(realne_y - (przyszle_y - 1)) < 3.7):
                return  # kolizja — nie opuszczaj
        if self.zejscie_y < 50:
            self.zejscie_y += 0.1

    def podniesc(self):
        """Klawisz C — podnoszenie."""
        if self.zejscie_y > 0:
            self.zejscie_y -= 0.1

    def rysuj(self, siatka_kontenerow, bujanie):
        """Rysuje chwytak i zmienia kolor wg stanu."""
        mozna_podniesc = False
        y_renderu      = 46
        chwytak_dol    = self.y - 1.5

        for kx, kz, ky in siatka_kontenerow:
            realne_y = 8.2 + bujanie + ky
            gora     = realne_y + 3.0
            if (abs((kx - 3) - self.x) < 2.0 and
                abs((kz - 1) - self.z) < 2.0 and
                chwytak_dol > gora - 1.0 and
                chwytak_dol < gora + 1.0):
                mozna_podniesc = True
                y_renderu = 46 + bujanie
                break

        if self.podniesiony_idx is not None:
            r, g, b = 1.0, 1.0, 0.0   # żółty — trzyma kontener
        elif mozna_podniesc:
            r, g, b = 0.0, 1.0, 0.2   # zielony — gotowy do złapania
        else:
            r, g, b = 0.9, 0.2, 0.6   # różowy — daleko

        rysuj_prostopadloscian(7.5, y_renderu, 28, 10, 3, 5.5, r, g, b)


class Suwnica:
    """Suwnica portowa z wózkiem i chwytakiem."""
    def __init__(self):
        self.x       = 0.0   # suwnica_x  (lewo-prawo, klawisze ←→)
        self.wyciag  = 0.0   # wyciag_z   (przód-tył,  klawisze Z/X)
        self.chwytak = Chwytak()

    def _aktualizuj_chwytak(self):
        self.chwytak.x = 7.5  + self.x
        self.chwytak.z = 28.0 + self.wyciag
        self.chwytak.y = 46.0 - self.chwytak.zejscie_y

    def obsluz_klawisze(self, keys, siatka_kontenerow, bujanie):
        self._aktualizuj_chwytak()
        ch = self.chwytak

        # ── lewo ──────────────────────────────────
        if keys[K_LEFT]:
            legalny = True
            px = ch.x + 0.05
            for kx, kz, ky in siatka_kontenerow:
                ry = 8.2 + bujanie + ky
                if (abs(kx - px) < 7.0 and
                    abs(kz - ch.z) < 6.2 and
                    abs(ry - (ch.y + 1.5)) < 3.7):
                    legalny = False
                    break
            if self.x < 70 and legalny:
                self.x += 0.05

        # ── prawo ─────────────────────────────────
        if keys[K_RIGHT]:
            legalny = True
            px = ch.x - 0.05
            for kx, kz, ky in siatka_kontenerow:
                ry = 8.2 + bujanie + ky
                if (abs(kx - px) < 7.0 and
                    abs(kz - ch.z) < 6.2 and
                    abs(ry - (ch.y + 1.5)) < 3.7):
                    legalny = False
                    break
            if self.x > -20 and legalny:
                self.x -= 0.05

        # ── wózek przód (Z) ───────────────────────
        if keys[K_z]:
            legalny = True
            pz = ch.z + 0.1
            for kx, kz, ky in siatka_kontenerow:
                ry = 8.2 + bujanie + ky
                if (abs(kx - ch.x) < 14.0 and
                    abs(kz - pz) < 6.9 and
                    abs(ry - (ch.y - 1.5)) < 4.345):
                    legalny = False
                    break
            if self.wyciag < 60 and legalny:
                self.wyciag += 0.1

        # ── wózek tył (X) ─────────────────────────
        if keys[K_x]:
            legalny = True
            pz = ch.z - 0.1
            for kx, kz, ky in siatka_kontenerow:
                ry = 8.2 + bujanie + ky
                if (abs(kx - ch.x) < 7.0 and
                    abs(kz - pz) < 6.2 and
                    abs(ry - (ch.y - 1.5)) < 3.7):
                    legalny = False
                    break
            if self.wyciag > -20 and legalny:
                self.wyciag -= 0.1

        # ── góra / dół ────────────────────────────
        if keys[K_v]:
            ch.opusc(siatka_kontenerow, bujanie)
        if keys[K_c]:
            ch.podniesc()

        self._aktualizuj_chwytak()

    def rysuj(self, siatka_kontenerow, bujanie, t):
        glPushMatrix()
        glTranslatef(self.x, 0, 0)

        glPushMatrix()
        glTranslatef(0, 0, self.wyciag)

        # Wyciąg i lina
        rysuj_prostopadloscian(7.5, 48, 28, 10, 3, 7.5, 0.6, 0.6, 0.6)
        for i in range(1, int(self.chwytak.zejscie_y * 10)):
            if i % 10 == 0:
                rysuj_prostopadloscian(7.5, 48 - (i / 10), 28, 0.3, 0.5, 0.3, 0, 0.9, 0.9)

        glPushMatrix()
        glTranslatef(0, -self.chwytak.zejscie_y, 0)
        self.chwytak.rysuj(siatka_kontenerow, bujanie)
        glPopMatrix()
        glPopMatrix()

        # Konstrukcja suwnicy
        rysuj_prostopadloscian(0,  0,  0,  1,   52,  1,   0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 0,  0,  1,   52,  1,   0.0, 0.0, 0.5)
        rysuj_prostopadloscian(0,  0,  25, 1,   52,  1,   0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 0,  25, 1,   52,  1,   0.0, 0.0, 0.5)

        rysuj_prostopadloscian(0,  10, 12.5, 1,   3,   26, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 10, 12.5, 1,   3,   26, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(0,  25, 12.5, 0.5, 0.5, 26, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(15, 25, 12.5, 0.5, 0.5, 26, 0.0, 0.0, 0.5)

        rysuj_prostopadloscian(0,  19, 6,  0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x= 45)
        rysuj_prostopadloscian(15, 19, 6,  0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x= 45)
        rysuj_prostopadloscian(0,  19, 19, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=-45)
        rysuj_prostopadloscian(15, 19, 19, 0.5, 0.5, 18, 0.0, 0.0, 0.5, rot_x=-45)

        rysuj_prostopadloscian(0,  37.5, 12, 0.5, 0.8, 35, 0.0, 0.0, 0.5, rot_x=45)
        rysuj_prostopadloscian(15, 37.5, 12, 0.5, 0.8, 35, 0.0, 0.0, 0.5, rot_x=45)

        rysuj_prostopadloscian(7.5, 0,  0,  16, 1, 1, 0.8, 0.3, 0.6)
        rysuj_prostopadloscian(7.5, 0,  25, 16, 1, 1, 0.8, 0.3, 0.6)

        for z_pos in range(30, 100, 20):
            rysuj_prostopadloscian(2,  49, z_pos, 1, 3, 10, 0.9, 0, 0)
            rysuj_prostopadloscian(13, 49, z_pos, 1, 3, 10, 0.9, 0, 0)
            if z_pos + 10 <= 90:
                rysuj_prostopadloscian(2,  49, z_pos + 10, 1, 3, 10, 1, 1, 1)
                rysuj_prostopadloscian(13, 49, z_pos + 10, 1, 3, 10, 1, 1, 1)

        kolor_swiatla = (1, 1, 1) if math.floor(t) % 4 == 0 else (0, 0, 0)
        rysuj_prostopadloscian(2,   51.8, 94.8, 0.4, 0.4, 0.4, *kolor_swiatla)
        rysuj_prostopadloscian(13,  51.8, 94.8, 0.4, 0.4, 0.4, *kolor_swiatla)
        rysuj_prostopadloscian(7.5, 80,   25,   2,   2,   2,   *kolor_swiatla)

        rysuj_prostopadloscian(2,  49, 12.5, 1.01, 3.01, 25, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(13, 49, 12.5, 1.01, 3.01, 25, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(7.5, 49, -30, 12, 3, 1, 0.0, 0.0, 0.5)

        rysuj_prostopadloscian(7.5, 49, 25, 16, 3, 1,  0.0, 0.0, 0.5)
        rysuj_prostopadloscian(7.5, 52, 0.5, 16, 3, 2, 0.0, 0.0, 0.5)

        rysuj_prostopadloscian(2,  49, -15, 1.01, 3.01, 30, 0.0, 0.0, 0.5)
        rysuj_prostopadloscian(13, 49, -15, 1.01, 3.01, 30, 0.0, 0.0, 0.5)

        rysuj_prostopadloscian(4.25,  66, 25, 29.5, 1, 1, 0.0, 0.0, 0.5, rot_z= 75.5)
        rysuj_prostopadloscian(10.75, 66, 25, 29.5, 1, 1, 0.0, 0.0, 0.5, rot_z=-75.5)
        rysuj_prostopadloscian(7.5,   67, 13, 1.5, 1.5, 35.5, 0.0, 0.0, 0.5, rot_x=-46)
        rysuj_prostopadloscian(7.5,   66, -2, 0.7, 0.7, 62,   0.0, 0.0, 0.5, rot_x=-28)

        rysuj_prostopadloscian(10.5, 66, 38, 0.3, 0.3, 39.5, 1, 0, 0, rot_x=47,    rot_y= 8)
        rysuj_prostopadloscian(4.5,  66, 38, 0.3, 0.3, 39.5, 1, 0, 0, rot_x=47,    rot_y=-8)
        rysuj_prostopadloscian(10.5, 66, 55, 0.3, 0.3, 66,   1, 0, 0, rot_x=25.75, rot_y= 5)
        rysuj_prostopadloscian(4.5,  66, 55, 0.3, 0.3, 66,   1, 0, 0, rot_x=25.75, rot_y=-5)

        rysuj_prostopadloscian(7.5, 50, -8, 10, 10, 15, 1, 1, 1)
        glPopMatrix()

class Tir:
    def __init__(self):
        self.id_tira=None
        try:
            self.id_tira = przygotuj_model(wczytaj_stl('./models/flatbed_truck.stl'))
        except:
            print("Nie znaleziono pliku flatbed_truck.stl!")
    def rysuj(self):
        rysuj_model(self.id_tira, 10, 0, 7,
                    0.5, 0.3, 0.1, skala=0.3, rot_x=-90, rot_z=90)

class Statek:       
    """Statek z kontenerami — generuje siatkę, kołysze się, rysuje."""
    def __init__(self):
        self.id_lodzi    = None
        self.id_kontenera = None
        self.bujanie     = 0.0

        # siatka i kolory — dokładnie jak w oryginale
        self.kolory           = []
        self.siatka_kontenerow = []
        for oy in WSPOLRZEDNE_Y:
            for ox in OFFSETY_X:
                for bx in BAZY_X:
                    for wz in WSPOLRZEDNE_Z:
                        self.kolory.append(random.choice(KOLORY_KONTENEROW))
                        self.siatka_kontenerow.append((bx - ox, wz, oy))

        try:
            self.id_lodzi = przygotuj_model(wczytaj_stl('./models/CargoShip.stl'))
        except:
            print("Nie znaleziono pliku CargoShip.stl!")

        try:
            self.id_kontenera = przygotuj_model(wczytaj_stl('./models/Container.stl'))
        except:
            print("Nie znaleziono pliku Container.stl!")

    def aktualizuj(self, t):
        self.bujanie = math.sin(t * 0.5) * STALA_FALOWANIA

    def rysuj(self, podniesiony_idx, chwytak_x, chwytak_y, chwytak_z, offset_y):
        if self.id_lodzi is None:
            return
        rysuj_model(self.id_lodzi, 30, -5 + self.bujanie, 80,
                    0.5, 0.3, 0.1, skala=0.8, rot_x=-90, rot_z=90)

        if self.id_kontenera is None:
            return
        for i, (k_x, k_z, k_y) in enumerate(self.siatka_kontenerow):
            if i == podniesiony_idx:
                render_x = chwytak_x + 5
                render_z = chwytak_z + 2
                render_y = chwytak_y + offset_y
            else:
                render_x = k_x
                render_z = k_z
                render_y = 8.2 + self.bujanie + k_y
            rysuj_model(
                self.id_kontenera,
                render_x, render_y, render_z,
                self.kolory[i][0], self.kolory[i][1], self.kolory[i][2],
                skala=0.24, rot_x=-90, rot_z=90,
            )


class Port:
    """Otoczenie — nabrzeże, lampy, woda, niebo."""
    def __init__(self):
        self.woda = Woda(1400, 1200, 200)

    def rysuj(self, t, kamera):
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        rysuj_prostopadloscian(0,  -30, -45, 200, 30, 150, 0.8, 0.8, 0.8)
        rysuj_prostopadloscian(0,    0,  28, 200,  2,   4, 0.6, 0.6, 0.6)
        rysuj_prostopadloscian(30,   0,   0, 120, 0.1,  1, 0.3, 0,   0)
        rysuj_prostopadloscian(30,   0,  25, 120, 0.1,  1, 0.3, 0,   0)

        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        lampka = -90
        for _ in range(20):
            rysuj_prostopadloscian(lampka, 2, 28, 0.7, 0.7, 0.7, 0.7, 0.4, 0)
            lampka += 10

        glPushMatrix()
        glTranslatef(7.5, -6, 600)
        self.woda.rysuj(t, 0.0, 0.3, 0.7)
        glPopMatrix()

        glPushMatrix()
        glLoadIdentity()
        glRotatef(kamera.rot_x, 1, 0, 0)
        glRotatef(kamera.rot_y, 0, 1, 0)
        rysuj_niebo_i_slonce()
        glPopMatrix()


# ═════════════════════════════════════════════
# GŁÓWNA PĘTLA
# ═════════════════════════════════════════════

def main():
    pygame.init()
    display = (1920, 1080)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    inicjalizuj_szescian()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, display[0] / display[1], 0.1, 1500.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    bg_color = [0.1, 0.1, 0.15, 1]
    glClearColor(*bg_color)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glLightfv(GL_LIGHT0, GL_POSITION, [50, 100, 50, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1])
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, bg_color)
    glFogf(GL_FOG_DENSITY, 0.002)
    glHint(GL_FOG_HINT, GL_NICEST)

    niebo_kolor = [0.2, 0.4, 0.8, 1]
    glClearColor(*niebo_kolor)
    glLightfv(GL_LIGHT0, GL_POSITION, [200, 300, -400, 1])
    glFogfv(GL_FOG_COLOR, niebo_kolor)

    # Obiekty sceny
    kamera  = Kamera()
    statek  = Statek()
    tir  = Tir()
    suwnica = Suwnica()
    port    = Port()

    clock = pygame.time.Clock()
    t     = 0.0

    while True:
        t += 0.05

        # ── Eventy ──────────────────────────────
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                pygame.quit()
                return
            if e.type == KEYDOWN and (e.key == K_RETURN or e.key == K_KP_ENTER):
                ch = suwnica.chwytak
                if ch.podniesiony_idx is None:
                    ch.probuj_zlapac(statek.siatka_kontenerow, statek.bujanie)
                else:
                    ch.upusc()

        # ── Mysz i klawisze ─────────────────────
        dx, dy = pygame.mouse.get_rel()
        kamera.obsluz_mysz(dx, dy)
        keys = pygame.key.get_pressed()
        kamera.obsluz_klawisze(keys)
        suwnica.obsluz_klawisze(keys, statek.siatka_kontenerow, statek.bujanie)

        # debug U
        if keys[K_u]:
            for i, (k_x, k_z, k_y) in enumerate(statek.siatka_kontenerow):
                kolor = statek.kolory[i] if i < len(statek.kolory) else (0, 0, 0)
                print(f"Kontener {i}: Pozycja X={k_x:.2f}, Z={k_z:.2f}, Y={k_y:.2f} | Kolor: {kolor}")

        # ── Aktualizacja ────────────────────────
        statek.aktualizuj(t)

        # ── Rysowanie ───────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        kamera.zastosuj()

        tir.rysuj()
        suwnica.rysuj(statek.siatka_kontenerow, statek.bujanie, t)

        ch = suwnica.chwytak
        statek.rysuj(ch.podniesiony_idx, ch.x, ch.y, ch.z, ch.podniesiony_offset)

        port.rysuj(t, kamera)

        pygame.display.flip()
        clock.tick(120)


if __name__ == "__main__":
    main()
