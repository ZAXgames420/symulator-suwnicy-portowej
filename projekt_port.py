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
# POMOCNICZE FUNKCJE OPENGL I UI
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


class RenderTekstu:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 14, bold=False)

    def rysuj_tekst(self, tekst, x, y, kolor=(255, 255, 255), male=False):
        f = self.font_small if male else self.font
        powierzchnia = f.render(tekst, True, kolor)
        
        try:
            dane_tekstury = pygame.image.tostring(powierzchnia, "RGBA", False)
        except ValueError:
            powierzchnia = powierzchnia.convert_alpha()
            dane_tekstury = pygame.image.tostring(powierzchnia, "RGBA", False)
            
        szerokosc, wysokosc = powierzchnia.get_size()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, szerokosc, wysokosc, 0, GL_RGBA, GL_UNSIGNED_BYTE, dane_tekstury)


        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 1.0) 
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + szerokosc, y)
        glTexCoord2f(1, 1); glVertex2f(x + szerokosc, y + wysokosc)
        glTexCoord2f(0, 1); glVertex2f(x, y + wysokosc)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex_id])

def wejdz_w_tryb_2d(szerokosc, wysokosc):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, szerokosc, wysokosc, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

def wyjdz_z_trybu_2d():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def rysuj_interfejs_2d(szerokosc, wysokosc, renderer, pokaz_pomoc):


    wejdz_w_tryb_2d(szerokosc, wysokosc)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


    if not pokaz_pomoc:
        box_w, box_h = 360, 50
        box_x = szerokosc - 600
        box_y = wysokosc - 200


        glColor4f(0.05, 0.05, 0.1, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(box_x, box_y)
        glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h)
        glVertex2f(box_x, box_y + box_h)
        glEnd()

        # Bardzo jasna, neonowa ramka (turkusowa) gwarantująca widoczność
        glColor4f(0.0, 0.9, 1.0, 0.9)
        glLineWidth(3)
        glBegin(GL_LINE_LOOP)
        glVertex2f(box_x, box_y)
        glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h)
        glVertex2f(box_x, box_y + box_h)
        glEnd()

        renderer.rysuj_tekst("Naciśnij [ F1 ], aby przeczytać instrukcję", box_x + 28, box_y + 13, kolor=(255, 255, 255), male=False)

    else:
        okno_w, okno_h = 560, 420
        okno_x = (szerokosc - okno_w) // 2
        okno_y = (wysokosc - okno_h) // 2

        glColor4f(0.05, 0.05, 0.1, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(okno_x, okno_y)
        glVertex2f(okno_x + okno_w, okno_y)
        glVertex2f(okno_x + okno_w, okno_y + okno_h)
        glVertex2f(okno_x, okno_y + okno_h)
        glEnd()

        glColor4f(0.0, 0.7, 0.9, 0.9)
        glLineWidth(3)
        glBegin(GL_LINE_LOOP)
        glVertex2f(okno_x, okno_y)
        glVertex2f(okno_x + okno_w, okno_y)
        glVertex2f(okno_x + okno_w, okno_y + okno_h)
        glVertex2f(okno_x, okno_y + okno_h)
        glEnd()

        y_offset = okno_y + 20
        renderer.rysuj_tekst("INSTRUKCJA SYMULATORA PORTU", okno_x + 70 + 30, y_offset, kolor=(0, 255, 255))
        
        y_offset += 40
        renderer.rysuj_tekst("[ TRYB AUTOMATYCZNY ]", okno_x + 70 + 30, y_offset, kolor=(255, 215, 0), male=True)
        y_offset += 25
        renderer.rysuj_tekst("Klawisz [ M ] - Włączenie / Wyłączenie pełnego automatu", okno_x + 70 + 50, y_offset, male=True)
        y_offset += 35
        renderer.rysuj_tekst("[ MANUALNE STEROWANIE SUWNICĄ ]", okno_x + 70 + 30, y_offset, kolor=(255, 215, 0), male=True)
        y_offset += 25
        renderer.rysuj_tekst("Strzałka w LEWO / PRAWO - Ruch suwnicy wzdłuż torów (Oś X)", okno_x + 70 + 50, y_offset, male=True)
        y_offset += 20
        renderer.rysuj_tekst("Klawisz [ Z ] / [ X ]     - Ruch wózka suwnicy (Oś Z)", okno_x + 70 + 50, y_offset, male=True)
        y_offset += 20
        renderer.rysuj_tekst("Klawisz [ V ] / [ C ]     - Opuszczanie / Podnoszenie chwytaka (Oś Y)", okno_x + 70 + 50, y_offset, male=True)
        y_offset += 20
        renderer.rysuj_tekst("Klawisz [ ENTER ]         - Chwytanie kontenera / Zrzut na Tira", okno_x + 70 + 50, y_offset, male=True)

        y_offset += 35
        renderer.rysuj_tekst("[ STEROWANIE KAMERĄ I OTOCZENIEM ]", okno_x + 70 + 30, y_offset, kolor=(255, 215, 0), male=True)
        y_offset += 25
        renderer.rysuj_tekst("Ruch Myszką               - Obracanie widoku kamery wokół osi", okno_x + 70 + 50, y_offset, male=True)
        y_offset += 20
        renderer.rysuj_tekst("Klawisze [ W ][ A ][ S ][ D ] - Poruszanie kamerą po porcie", okno_x + 70 + 50, y_offset, male=True)
        y_offset += 20
        renderer.rysuj_tekst("Klawisz [ SPACJA ] / [ LSHIFT ] - Lot kamery w Górę / w Dół", okno_x + 70 + 50, y_offset, male=True)
        y_offset += 20
        renderer.rysuj_tekst("Klawisz [ R ]             - Resetowanie kontenerów i pozycji statku", okno_x + 70 + 50, y_offset, male=True)

        y_offset += 35
        renderer.rysuj_tekst("Naciśnij ponownie [ F1 ] aby zamknąć to okno", okno_x + 70 + 120, y_offset, kolor=(0, 255, 255), male=True)

    wyjdz_z_trybu_2d()

# ═════════════════════════════════════════════
# KLASY MODELU TRÓJWYMIAROWEGO
# ═════════════════════════════════════════════

class Woda:
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
    def __init__(self):
        self.zejscie_y = 0.0
        self.x = 7.5
        self.y = 46.0
        self.z = 28.0
        self.podniesiony_idx    = None
        self.podniesiony_offset = 0.0

    def probuj_zlapac(self, siatka_kontenerow, bujanie):
        chwytak_dol = self.y - 1.5
        for idx, (kx, kz, ky) in enumerate(siatka_kontenerow):
            realne_y = 8.2 + bujanie + ky
            gora     = realne_y + 3.0
            if (abs((kx - 5) - self.x) < 1.0 and
                abs((kz - 2) - self.z) < 1.0 and
                chwytak_dol > gora - 1.0 and
                chwytak_dol < gora + 1.0):
                self.podniesiony_idx    = idx
                self.podniesiony_offset = -(1.5 + 3.0)
                return

    def upusc(self):
        self.podniesiony_idx    = None
        self.podniesiony_offset = 0.0

    def opusc(self, siatka_kontenerow, bujanie):
        przyszle_y = self.y - 0.1
        for kx, kz, ky in siatka_kontenerow:
            realne_y = 8.2 + bujanie + ky
            if (abs(kx - self.x) < 7.0 and
                abs(kz - self.z) < 6.2 and
                abs(realne_y - (przyszle_y - 1)) < 3.7):
                return
        if self.zejscie_y < 50:
            self.zejscie_y += 0.1

    def podniesc(self):
        if self.zejscie_y > 0:
            self.zejscie_y -= 0.1

    def rysuj(self, siatka_kontenerow, bujanie, tir=None):
        mozna_podniesc = False
        y_renderu      = 46
        chwytak_dol    = self.y - 1.5

        for kx, kz, ky in siatka_kontenerow:
            realne_y = 8.2 + bujanie + ky
            gora     = realne_y + 3.0
            if (abs((kx - 5) - self.x) < 1.0 and
                abs((kz - 2) - self.z) < 1.0 and
                chwytak_dol > gora - 1.0 and
                chwytak_dol < gora + 1.0):
                mozna_podniesc = True
                y_renderu = 46 + bujanie
                break

        # POPRAWKA: Chwytak świeci na turkusowo tylko wtedy, gdy jest nad tirem I odpowiednio nisko
        mozna_odlozyc_na_tira = False
        if self.podniesiony_idx is not None and tir is not None:
            render_x = self.x + 5
            render_z = self.z + 2
            if tir.czy_kontener_trafia(render_x, render_z) and self.zejscie_y >= 32.0:
                mozna_odlozyc_na_tira = True

        if mozna_odlozyc_na_tira:
            r, g, b = 0.0, 1.0, 1.0
        elif self.podniesiony_idx is not None:
            r, g, b = 1.0, 1.0, 0.0
        elif mozna_podniesc:
            r, g, b = 0.0, 1.0, 0.2
        else:
            r, g, b = 0.9, 0.2, 0.6

        rysuj_prostopadloscian(7.5, y_renderu, 28, 10, 3, 5.5, r, g, b)

class Suwnica:
    def __init__(self):
        self.x       = 0.0
        self.wyciag  = 0.0
        self.chwytak = Chwytak()

    def _aktualizuj_chwytak(self):
        self.chwytak.x = 7.5  + self.x
        self.chwytak.z = 28.0 + self.wyciag
        self.chwytak.y = 46.0 - self.chwytak.zejscie_y

    def obsluz_klawisze(self, keys, siatka_kontenerow, bujanie):
        self._aktualizuj_chwytak()
        ch = self.chwytak

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

        if keys[K_v]:
            ch.opusc(siatka_kontenerow, bujanie)
        if keys[K_c]:
            ch.podniesc()

        self._aktualizuj_chwytak()

    def rysuj(self, siatka_kontenerow, bujanie, t, tir=None):
        glPushMatrix()
        glTranslatef(self.x, 0, 0)

        glPushMatrix()
        glTranslatef(0, 0, self.wyciag)

        rysuj_prostopadloscian(7.5, 48, 28, 10, 3, 7.5, 0.6, 0.6, 0.6)
        for i in range(1, int(self.chwytak.zejscie_y * 10)):
            if i % 10 == 0:
                rysuj_prostopadloscian(7.5, 48 - (i / 10), 28, 0.3, 0.5, 0.3, 0, 0.9, 0.9)

        glPushMatrix()
        glTranslatef(0, -self.chwytak.zejscie_y, 0)
        self.chwytak.rysuj(siatka_kontenerow, bujanie, tir)
        glPopMatrix()
        glPopMatrix()

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
    X_START  = -180.0
    X_STOP   =   10.0
    X_WYJAZD =  200.0
    Z_POS    =    7.0
    Y_POS    =    0.0
    SKALA    =    0.3
    PRED_WJAZD  = 1.2
    PRED_WYJAZD = 1.8
    FADE_SPEED  = 1.5

    def __init__(self):
        self.id_tira = None
        try:
            self.id_tira = przygotuj_model(wczytaj_stl('./models/flatbed_truck.stl'))
        except:
            print("Nie znaleziono pliku flatbed_truck.stl!")

        self.x     = self.X_START
        self.stan  = 'wjazd'
        self.timer = 0.0
        self.alpha = 0.0
        self.ma_kontener    = False
        self.kontener_kolor = (1.0, 0.2, 0.1)
        self.kontener_id    = None

    def aktualizuj(self, dt):
        if self.stan == 'wjazd':
            self.x    += self.PRED_WJAZD
            self.alpha = min(1.0, self.alpha + self.FADE_SPEED * dt)
            if self.x >= self.X_STOP:
                self.x    = self.X_STOP
                self.stan = 'stoi'

        elif self.stan == 'stoi':
            self.alpha = min(1.0, self.alpha + self.FADE_SPEED * dt)

        elif self.stan == 'czeka':
            self.timer -= dt
            if self.timer <= 0.0:
                self.stan = 'wyjazd'

        elif self.stan == 'wyjazd':
            self.x    += self.PRED_WYJAZD
            self.alpha = max(0.0, self.alpha - self.FADE_SPEED * dt)
            if self.x >= self.X_WYJAZD:
                self.x              = self.X_START
                self.stan           = 'wjazd'
                self.alpha          = 0.0
                self.ma_kontener    = False
                self.kontener_kolor = random.choice(KOLORY_KONTENEROW)

    def poloz_kontener(self, kolor):
        if self.stan != 'stoi':
            return False
        self.kontener_kolor = kolor
        self.ma_kontener    = True
        self.stan           = 'czeka'
        self.timer          = 1.0
        return True

    def czy_kontener_trafia(self, render_x, render_z):
        return (self.stan == 'stoi' and
                abs(render_x - self.x) < 20.0 and
                abs(render_z - self.Z_POS) < 10.0)

    def rysuj(self):
        if self.id_tira is None or self.alpha <= 0.0:
            return

        a = self.alpha
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glPushMatrix()
        glTranslatef(self.x, self.Y_POS, self.Z_POS)
        glRotatef(-90, 1, 0, 0)
        glRotatef(90,  0, 0, 1)
        glScalef(self.SKALA, self.SKALA, self.SKALA)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.5, 0.3, 0.1, a])
        glColor4f(0.5, 0.3, 0.1, a)
        glCallList(self.id_tira)
        glPopMatrix()

        if self.ma_kontener and self.kontener_id is not None:
            r, g, b = self.kontener_kolor
            glPushMatrix()
            glTranslatef(self.x, 4.5, 13.5)
            glRotatef(-90, 1, 0, 0)
            glRotatef(90,  0, 0, 1)
            glScalef(0.24, 0.24, 0.24)
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [r, g, b, a])
            glColor4f(r, g, b, a)
            glCallList(self.kontener_id)
            glPopMatrix()

class Statek:       
    def __init__(self):
        self.id_lodzi    = None
        self.id_kontenera = None
        self.bujanie     = 0.0

        self.kolory           = []
        self.siatka_kontenerow = []
        self.generuj_kontenery()

        try:
            self.id_lodzi = przygotuj_model(wczytaj_stl('./models/CargoShip.stl'))
        except:
            print("Nie znaleziono pliku CargoShip.stl!")

        try:
            self.id_kontenera = przygotuj_model(wczytaj_stl('./models/Container.stl'))
        except:
            print("Nie znaleziono pliku Container.stl!")

    def generuj_kontenery(self):
        self.kolory.clear()
        self.siatka_kontenerow.clear()
        for oy in WSPOLRZEDNE_Y:
            for ox in OFFSETY_X:
                for bx in BAZY_X:
                    for wz in WSPOLRZEDNE_Z:
                        self.kolory.append(random.choice(KOLORY_KONTENEROW))
                        self.siatka_kontenerow.append((bx - ox, wz, oy))

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
# GŁÓWNA PĘTLA Z MASZYNĄ STANÓW (FSM) ORAZ UI
# ═════════════════════════════════════════════

def znajdz_najwyzszy_kontener(siatka):
    if not siatka: return None
    najwyzszy_idx = 0
    max_y = -999.0
    for i, (kx, kz, ky) in enumerate(siatka):
        if ky > max_y:
            max_y = ky
            najwyzszy_idx = i
    return najwyzszy_idx

def main():
    pygame.init()
    display = (1920, 1080)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    inicjalizuj_szescian()
    renderer_tekstu = RenderTekstu()

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

    kamera  = Kamera()
    statek  = Statek()
    tir     = Tir()
    tir.kontener_id = statek.id_kontenera
    suwnica = Suwnica()
    port    = Port()

    clock = pygame.time.Clock()
    t     = 0.0

    pokaz_pomoc = False
    tryb_auto = False
    auto_stan = 0
    target_suwnica_x = 0.0
    target_suwnica_wyciag = 0.0


    PREDKOSC_AUTO_X = 0.05
    PREDKOSC_AUTO_Z = 0.1

    while True:
        dt = clock.get_time() / 1000.0
        t += 0.05

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                pygame.quit()
                return
            
            if e.type == KEYDOWN:
                if e.key == K_F1:
                    pokaz_pomoc = not pokaz_pomoc

                if e.key == K_RETURN or e.key == K_KP_ENTER:
                    if not tryb_auto:
                        ch = suwnica.chwytak
                        if ch.podniesiony_idx is None:
                            ch.probuj_zlapac(statek.siatka_kontenerow, statek.bujanie)
                        else:
                            render_x = ch.x + 5
                            render_z = ch.z + 2
                            
                            # POPRAWKA: Sprawdzamy czy trafia w Tira ORAZ czy wysokość chwytaka (zejscie_y) jest odpowiednio niska (tuż nad naczepą)

                            if tir.czy_kontener_trafia(render_x, render_z) and ch.zejscie_y >= 37.0:
                                kolor = statek.kolory[ch.podniesiony_idx]
                                if tir.poloz_kontener(kolor):
                                    statek.siatka_kontenerow.pop(ch.podniesiony_idx)
                                    statek.kolory.pop(ch.podniesiony_idx)

                                    ch.upusc() # Przeniesione tutaj - puszcza tylko po udanym zrzucie na ciężarówkę

                if e.key == K_r:
                    suwnica.chwytak.upusc()
                    statek.generuj_kontenery()
                    tryb_auto = False

                if e.key == K_m:
                    tryb_auto = not tryb_auto
                    auto_stan = 0

        dx, dy = pygame.mouse.get_rel()
        kamera.obsluz_mysz(dx, dy)
        keys = pygame.key.get_pressed()
        kamera.obsluz_klawisze(keys)

        if keys[K_u]:
            for i, (k_x, k_z, k_y) in enumerate(statek.siatka_kontenerow):
                kolor = statek.kolory[i] if i < len(statek.kolory) else (0, 0, 0)
                print(f"Kontener {i}: Pozycja X={k_x:.2f}, Z={k_z:.2f}, Y={k_y:.2f} | Kolor: {kolor}")

        # ─────────────────────────────────────────────
        # LOGIKA STEROWANIA AUTOPILOTA (FSM)
        # ─────────────────────────────────────────────
        if not tryb_auto:
            suwnica.obsluz_klawisze(keys, statek.siatka_kontenerow, statek.bujanie)
        else:
            if auto_stan == 0: 
                ch = suwnica.chwytak
                if ch.podniesiony_idx is not None:
                    auto_stan = 5
                else:
                    idx = znajdz_najwyzszy_kontener(statek.siatka_kontenerow)
                    if idx is not None:
                        kx, kz, ky = statek.siatka_kontenerow[idx]
                        target_suwnica_x = (kx - 3) - 10
                        target_suwnica_wyciag = (kz - 1) - 29.0
                        auto_stan = 1
                    else:
                        tryb_auto = False

            elif auto_stan == 1: 
                if suwnica.chwytak.zejscie_y > 0.05:
                    suwnica.chwytak.podniesc()
                else:
                    suwnica.chwytak.zejscie_y = 0.0
                    auto_stan = 2

            elif auto_stan == 2: 
                dojechal_x = False
                dojechal_z = False
                
                diff_x = target_suwnica_x - suwnica.x
                if abs(diff_x) < PREDKOSC_AUTO_X:
                    suwnica.x = target_suwnica_x
                    dojechal_x = True
                else:
                    suwnica.x += PREDKOSC_AUTO_X if diff_x > 0 else -PREDKOSC_AUTO_X

                diff_z = target_suwnica_wyciag - suwnica.wyciag
                if abs(diff_z) < PREDKOSC_AUTO_Z:
                    suwnica.wyciag = target_suwnica_wyciag
                    dojechal_z = True
                else:
                    suwnica.wyciag += PREDKOSC_AUTO_Z if diff_z > 0 else -PREDKOSC_AUTO_Z

                if dojechal_x and dojechal_z:
                    auto_stan = 3

            elif auto_stan == 3: 
                suwnica.chwytak.opusc(statek.siatka_kontenerow, statek.bujanie)
                suwnica._aktualizuj_chwytak()
                suwnica.chwytak.probuj_zlapac(statek.siatka_kontenerow, statek.bujanie)
                if suwnica.chwytak.podniesiony_idx is not None:
                    auto_stan = 5

            elif auto_stan == 5: 
                if suwnica.chwytak.zejscie_y > 0.05:
                    suwnica.chwytak.podniesc()
                else:
                    suwnica.chwytak.zejscie_y = 0.0
                    auto_stan = 6

            elif auto_stan == 6: 
                if tir.stan == 'stoi':
                    target_suwnica_x = 0.0
                    target_suwnica_wyciag = -16.5
                    auto_stan = 7

            elif auto_stan == 7: 
                dojechal_x = False
                dojechal_z = False
                
                diff_x = target_suwnica_x - suwnica.x
                if abs(diff_x) < PREDKOSC_AUTO_X:
                    suwnica.x = target_suwnica_x
                    dojechal_x = True
                else:
                    suwnica.x += PREDKOSC_AUTO_X if diff_x > 0 else -PREDKOSC_AUTO_X

                diff_z = target_suwnica_wyciag - suwnica.wyciag
                if abs(diff_z) < PREDKOSC_AUTO_Z:
                    suwnica.wyciag = target_suwnica_wyciag
                    dojechal_z = True
                else:
                    suwnica.wyciag += PREDKOSC_AUTO_Z if diff_z > 0 else -PREDKOSC_AUTO_Z

                if dojechal_x and dojechal_z:
                    auto_stan = 8

            elif auto_stan == 8: 
                if suwnica.chwytak.zejscie_y < 38.0:
                    suwnica.chwytak.zejscie_y += 0.1
                else:
                    auto_stan = 9

            elif auto_stan == 9: 
                ch = suwnica.chwytak
                render_x = ch.x + 5
                render_z = ch.z + 2
                if tir.czy_kontener_trafia(render_x, render_z):
                    kolor = statek.kolory[ch.podniesiony_idx]
                    if tir.poloz_kontener(kolor):
                        statek.siatka_kontenerow.pop(ch.podniesiony_idx)
                        statek.kolory.pop(ch.podniesiony_idx)
                        ch.upusc()
                        auto_stan = 10
                elif tir.stan != 'stoi':
                    auto_stan = 6 

            elif auto_stan == 10: 
                if suwnica.chwytak.zejscie_y > 0.05:
                    suwnica.chwytak.podniesc()
                else:
                    suwnica.chwytak.zejscie_y = 0.0
                    auto_stan = 0

            suwnica._aktualizuj_chwytak()

        statek.aktualizuj(t)
        tir.aktualizuj(dt)

        # RENDEROWANIE SCENY 3D
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        kamera.zastosuj()

        tir.rysuj()
        suwnica.rysuj(statek.siatka_kontenerow, statek.bujanie, t, tir)

        ch = suwnica.chwytak
        statek.rysuj(ch.podniesiony_idx, ch.x, ch.y, ch.z, ch.podniesiony_offset)
        port.rysuj(t, kamera)


        # RENDEROWANIE INTERFEJSU 2D
        rysuj_interfejs_2d(display[0], display[1], renderer_tekstu, pokaz_pomoc)

        pygame.display.flip()
        clock.tick(120)


if __name__ == "__main__":
    main()