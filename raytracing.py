#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def create_Ray(O, D):
    # Cette fonction cree une ray
    ray={
        "origine":O,
        "direction":D
        }
    return ray

def create_sphere(P, r, ambient, diffuse, specular, reflection, i):
    # TODO: Description de a fonction
    sphere={"type":"sphere",
            "position":np.array(P),
            "radius":r,
            "ambient":np.array(ambient), # Coleur de base
            "diffuse":np.array(diffuse), # Couleur utilise avec une lumiere diffuse
            # TODO: Add description for speculaire
            "specular":np.array(specular), # Couleur speculaire
            "index":i,
            "reflection":reflection
            }
    return sphere
    
def create_plane(P, n, ambient, diffuse, specular, reflection, i):
    # TODO: Description de a fonction
    plane={"type":"plan","position":P,
           "normale":n,
           "ambient":np.array(ambient), # Coleur de base
           "diffuse":np.array(diffuse), # Couleur utilise avec une lumiere diffuse
           # TODO: Add description for speculaire
           "specular":np.array(specular), # Couleur speculaire
           "index": i,
           "reflection":reflection
           }
    return plane

def normalize(x):
    # TODO: Write simple desctiption of this function
    return x / np.linalg.norm(x)

def rayAt(ray,t):
    # TODO: Write simple desctiption of this function
    D = ray["direction"]
    O = ray["origine"]
    return O + t * D

def get_Normal(obj, M):
    # TODO: Write simple desctiption of this function
    if obj["type"] == "sphere":
        vect_normal = M - obj['position']
        return normalize(vect_normal)
    if obj["type"]=="plan":
        vect_normal = obj['normale']
        return normalize(vect_normal)

def intersect_Plane(ray, plane):
    # TODO: Description de la fonction
    D = ray["direction"]
    O = ray["origine"]
    P = plane["position"]
    N = plane["normale"]
    # Distance entre deux points
    t = (-np.dot((O - P), N)) / (np.dot(D, N))
    a = abs(np.dot(D, N))
    b = 10 ** (-6)
    if t > 0:
        if a > b:
            return t
        else:
            return np.inf
    else:
        return np.inf

def intersect_Sphere(ray, sphere):
    # TODO: Description de la fonction
    D = ray["direction"]
    O = ray["origine"]
    S = sphere["position"]
    R = sphere["radius"]
    OS = O - S
    a = np.dot(D,D)
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R ** 2
    delta = b ** 2 - 4 * a * c
    if delta > 0:
        t1 = (-b + delta ** 0.5) / (2 * a)
        t2 = (-b - delta ** 0.5) / (2 * a)
        if t1 > 0 and t2 > 0:    
            return min(t1,t2)
        elif t1 > 0:
            return t1
        elif t2 > 0:
            return t2
    else:
        return np.inf
    return np.inf

def intersect_Scene(ray, obj):
    # TODO: Description de la fonction
    if obj["type"]=="sphere":
        return intersect_Sphere(ray, obj)
    elif obj["type"]=="plan":
        return intersect_Plane(ray, obj)

def Is_in_Shadow(obj_min,P,N):
    # TODO: Description de la fonction
    liste = []
    lpl = normalize(Light["position"] - P)
    Pe = P + acne_eps*lpl
    ray_test = create_Ray(Pe, lpl)
    for i in scene:
        if i["index"] != obj_min["index"]:
            t = intersect_Scene(ray_test, i)
            if t != np.inf:
                liste.append(i)
    if liste==[]:
        return True  #Non in shadow
    else:
        return False #Is in shadow

def eclairage(obj, light, P):
    # TODO: Description de la fonction
    # obj: dictionaire avec la description de l'objet
    # ligth: dictionaire representant la source lumineuse
    # P: repesentant position du point de l'objet pour lequelle on veut calculer
    # la couleur
    # import global variables
    global C, materialShininess
    L_pos = light["position"] # Position de la source de lumiere
    l = normalize(L_pos - P) # Light direction
    ka = obj["ambient"]   # coulour ambiante de l'objet
    la = light["ambient"] # couleur ambiante de la lumiere
    kd = obj["diffuse"]   # Couloeur diffuse de 'lobjet
    ld = light["diffuse"] # couleur diffuse de la lumiere
    N = get_Normal(obj, P)
    ks = obj["specular"]  # couleur speculaire lumiere
    ls = light["specular"] # couleur speculaire de la lumiere
    B = materialShininess
    c = normalize(C - P)
    # Coleur Diffuse
    cd = ka * la + kd * ld * np.dot(l, N) 
    cd = list(cd)
    liste = []
    for i in cd:
        if i < 0:
            liste.append(0.0)
        else:
            liste.append(i)
    # Couleu diffuse
    cd = np.array(liste)
    # pvect est le produit scalaire du couleur total 
    pvect = np.dot(normalize(l + c), N)
    # Si le produit scalaire est en dessous de 0 on renvoie une couleur noire
    if pvect < 0:
        pvect = 0
    # Coleur Total
    ct = cd + (ks * ls * ((pvect) ** (B / 4.0)))
    if not(pvect < 0):
        pass
    ct = list(ct)
    liste = []
    for i in ct:
        if i < 0:
            liste.append(0)
        else:
            liste.append(i)
    #ct = np.array(liste)
    return ct

def reflected_ray(dirRay,N):
    return dirRay - 2 * np.dot(dirRay,N)*N

def compute_reflection(rayTest,depth_max,col):
    # TODO: Write simple desctiption of this function
    for i in range(1, depth_max):
        obji, Pi, Ni, col_rayi = trace_ray(rayTest)
        if obji != None:
            c = obji["reflection"]
            rayTest["origine"] = Pi + Ni*acne_eps
            rayTest["direction"] = reflected_ray(rayTest["direction"],Ni)
            col = (col + c*obji["ambient"])/(1+c)
        else:
            return col
    return col

def trace_ray(ray):
    # TODO: Faire description de la fonction
    global scene, Light # Use scene glbal variable
    D = ray["direction"]
    O = ray["origine"]
    distance_vector = np.inf # distance par defaut cést l'infinie
    
    # Loop on all objects
    for obj_test in scene:
        # trouver lintersection des rayons
        distance_dintersection = intersect_Scene(ray,obj_test)
        if distance_dintersection < distance_vector:
            # Si il y a une intersecion fait ca
            distance_vector = distance_dintersection
            obj = obj_test

    if distance_vector == np.inf:
        return None, None, None, None    
    else:
        # Intersection between object et ray
        P = distance_vector * D + O
        # Normal vector between object and P
        N = get_Normal(obj, P)
        # Get object color
        coleur_ray = eclairage(obj, Light, P)
        # TODO: Uncomment next images to continue workshop
#         if Is_in_Shadow(obj,P,N):
#             P = O + D*distance
#             N = get_Normal(obj, P)
#             col_ray = eclairage(obj,Light,P)
#         else: # si la couleur est noire renvoyer noire et pas de couleurs negatifs
#             col_ray = np.array([0,0,0])
    return obj, P, N, coleur_ray

# Taille de l'image
w = 800
h = 600
acne_eps = 1e-4
materialShininess = 50

img = np.zeros((h, w, 3)) # image vide : que du noir
#Aspect ratio
r = float(w) / h
# coordonnées de l'écran : x0, y0, x1, y1.
S = (-1., -1. / r , 1., 1. / r )

# Position et couleur de la source lumineuse
Light = { 'position': np.array([5, 5, 0]),
          'ambient': np.array([0.05, 0.05, 0.05]),
          'diffuse': np.array([1, 1, 1]),
          'specular': np.array([1.0, 1.0, 1.0]) }

L = Light['position']

col = np.array([0.2, 0.2, 0.7])  # couleur de base
C = np.array([0., 0.1, 1.1])  # Coordonée du centre de la camera.
Q = np.array([0,0.3,0])  # Orientation de la caméra
img = np.zeros((h, w, 3)) # image vide : que du noir
materialShininess = 50
skyColor = np.array([0.321, 0.752, 0.850])
whiteColor = np.array([1,1,1])
depth_max = 10

# Scene Description
scene = [create_sphere([.75, -.3, -1.], # Position
                         .6, # Rayon
                         np.array([1. , 0.6, 0. ]), # couleur ambiant
                         np.array([1. , 0.6, 0. ]), # couleur diffuse
                         np.array([1.0, 1.0, 1.0]), # specular
                         0.2, # reflection index
                         1), # index
          create_plane([0., -.9, 0.], # Position
                         [0, 1, 0], # Normal
                         np.array([0.145, 0.584, 0.854]), # couleur ambiant
                         np.array([0.145, 0.584, 0.854]), # couleur diffuse
                         np.array([1.0, 1.0, 1.0]), # specular
                         0.7, # reflection index
                         2), # index
         # If you wanna add new shapes tp scene you add them here
         #create_plane(),
         #create_ssphere(),
         #create_cone(),
         ]

# Following condition only executes when this file is executed, so now
# you can import this file functions to make differents scenes on different
# files
if __name__ == "__main__":
    # Loop through all pixels.
    for i, x in enumerate(np.linspace(S[0], S[2], w)):
        # TODO: Remove next two lines to improve speed
        if i % 10 == 0:
            print(F"{i / float(w) * 100}")
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            direction_rayon = normalize(np.array([x ,y ,0]) - C)
            raytest = create_Ray(C, direction_rayon)
            obj, P, N, col_ray  = trace_ray(raytest)
            #col_ray = compute_reflection(raytest, depth_max, col_ray)
            # Recupere la coleur du point dintersection et le rajouer a celle de P
            # Associer la coleur du point dintersection au pixel xorrespondant a l'image img
            try:
                img[h - j - 1, i, :] = np.clip(col_ray, 0, 1) # la fonction clip permet de "forcer" col a être dans [0,1]
            except TypeError:
                # dans le cas ou col_ray est None c'es ignore
                pass
            except Exception as e:
                print(F"Erorr => {e.__class__.__name__} ==> {e}")       

    plt.imsave('figRaytracing.png', img)
    #plt.imshow(img)