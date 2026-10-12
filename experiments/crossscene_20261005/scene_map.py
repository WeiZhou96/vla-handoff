# -*- coding: utf-8 -*-
"""Map initial-condition coordinates of CALVIN scene D to the furniture layouts of scenes A/B/C (semantics unchanged: block on the table / in the sliding cabinet left / right).
Based on the measured positions of slide_link and button_link in each scene (layout_probe.json):
  positions inside the sliding cabinet: in D, slider_left = slide_x + (-0.0009), slider_right = slide_x + 0.3103;
               in scenes where the sliding cabinet is on the right side of the table (slide_x > 0), the mirrored values slide_x - 0.0009*(-1) and slide_x - 0.3103 are used.
  table positions: x = 0.05 and 0.23 in D (y = -0.12); if a position is closer than 0.10 m to the button, its mirror image about x = 0 is used instead.
Joint states (sliding door, drawer, button, switch) and block orientations are unchanged."""
import numpy as np

LAYOUT = {"calvin_scene_D": dict(slide_x=-0.24, button_x=-0.12), "calvin_scene_A": dict(slide_x=-0.24, button_x=-0.28),
          "calvin_scene_B": dict(slide_x=0.228, button_x=0.28), "calvin_scene_C": dict(slide_x=0.20, button_x=-0.12)}
D_SL, D_SR = np.array([-2.40851662e-01, 9.24044687e-02]), np.array([7.03416330e-02, 9.24044687e-02])
D_T = [np.array([5.00000896e-02, -1.20000177e-01]), np.array([2.29995412e-01, -1.19995140e-01])]


def map_scene_obs(scene_obs, scene):
    if scene in (None, "calvin_scene_D"):
        return np.asarray(scene_obs, np.float64).copy()
    L = LAYOUT[scene]; sd = LAYOUT["calvin_scene_D"]["slide_x"]
    s = 1.0 if L["slide_x"] < 0 else -1.0
    sl = np.array([L["slide_x"] + s * (D_SL[0] - sd), D_SL[1]]); sr = np.array([L["slide_x"] + s * (D_SR[0] - sd), D_SR[1]])
    tab = []
    for t in D_T:
        tab.append(np.array([-t[0], t[1]]) if abs(t[0] - L["button_x"]) < 0.10 else t.copy())
    sc = np.asarray(scene_obs, np.float64).copy()
    for b in (6, 12, 18):
        xy = sc[b:b + 2]
        if np.linalg.norm(xy - D_SL) < 0.02:
            sc[b:b + 2] = sl
        elif np.linalg.norm(xy - D_SR) < 0.02:
            sc[b:b + 2] = sr
        elif np.linalg.norm(xy - D_T[0]) < 0.02:
            sc[b:b + 2] = tab[0]
        elif np.linalg.norm(xy - D_T[1]) < 0.02:
            sc[b:b + 2] = tab[1]
    return sc
