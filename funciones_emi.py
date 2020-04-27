# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 01:17:26 2019

@author: Emiliano Kalafatic
"""

import numpy as np
#import math
import cv2 as cv
from matplotlib import pyplot as plt
import restauracion as rst
import os

def nothing(x):
  pass

def numbers_to_strings(argument): 
    switcher = { 
        0: "zero", 
        1: "one", 
        2: "two", 
    } 
  
    # get() method of dictionary data type returns  
    # value of passed argument if it is present  
    # in dictionary otherwise second argument will 
    # be assigned as default value of passed argument 
    return switcher.get(argument, "nothing") 


# =============================================================================
# #############################################################################
# #############################################################################
# #----------------------------- GRAFICACIÓN ----------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
def graficar_plt(img1,title1='',nfigure=1,mapa='gray'):
    plt.figure(nfigure)
    plt.imshow(img1,cmap=mapa),plt.title(title1)
    plt.show()
    
def graficar_plt2(img1,img2,title1='',title2='',nfigure=1,mapa1='gray',mapa2='gray'):
    plt.figure(nfigure)
    plt.subplot(121),plt.imshow(img1,cmap=mapa1),plt.title(title1)
    plt.subplot(122),plt.imshow(img2,cmap=mapa2),plt.title(title2)
    plt.show()
    
def graficar_plt3(img1,img2,img3,title1='',title2='',title3='',nfigure=1, mapa1='gray',mapa2='gray', mapa3='gray'):
    plt.figure(nfigure)
    plt.subplot(131),plt.imshow(img1,cmap=mapa1),plt.title(title1)
    plt.subplot(132),plt.imshow(img2,cmap=mapa2),plt.title(title2)
    plt.subplot(133),plt.imshow(img3,cmap=mapa3),plt.title(title3)
    plt.show()
    
def graficar_plt4(img1,img2,img3,img4,title1='',title2='',title3='',title4='',nfigure=1,mapa1='gray',mapa2='gray', mapa3='gray', mapa4='gray'):
    plt.figure(nfigure)
    plt.subplot(221),plt.imshow(img1,cmap=mapa1),plt.title(title1)
    plt.subplot(222),plt.imshow(img2,cmap=mapa2),plt.title(title2)
    plt.subplot(223),plt.imshow(img3,cmap=mapa3),plt.title(title3)
    plt.subplot(224),plt.imshow(img4,cmap=mapa4),plt.title(title4)
    plt.show()

# =============================================================================
# CARGAR CONJUNTO IMÁGENES de un directorio
# =============================================================================
def load_images(folder):    # OJO las imágenes deben ser en jpg
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))
        img = cv.cvtColor(img,cv.COLOR_RGB2BGR).astype(np.uint8)
        if img is not None:
            images.append(img)
    return np.array(images)


# Limpiar bien la imagen previamente para que queden los objetos solos.
def convex_hull(img_bin):
    imagen_contornos, contornos, jerarquia = cv.findContours(img_bin,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    # create hull array for convex hull points
    hull = []
     
    # calculate points for each contour
    for i in range(len(contornos)):
        # creating convex hull object for each contour
        hull.append(cv.convexHull(contornos[i], False))
        
    # create an empty black image
    drawing = np.zeros((img_bin.shape[0], img_bin.shape[1], 3), np.uint8)
     
    # draw contours and hull points
    for i in range(len(contornos)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv.drawContours(drawing, contornos, i, color_contours, 1, 8, jerarquia)
        # draw ith convex hull object
        cv.drawContours(drawing, hull, i, color, 1, 8)
    
    return drawing


def graficar(img,maximo,minimo,mapa,titulo='',ejes="off"):
#    ventana = plt.figure()
    # ventana.canvas.set_window_title(titulo)
    plt.axis(ejes)
    plt.imshow(img, vmax=maximo, vmin=minimo, cmap=mapa)
    plt.title(titulo)
    
def histograma(img, title="Histograma"):
    plt.figure()
    plt.hist(img.flatten(), 255)
    plt.title(title)
    
def histogramaRGB(img):
    color = ('blue','green','red')
    plt.figure()
    for i,col in enumerate(color):
        hist = cv.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color = col)
        plt.xlim([0,256])
    plt.title("histograma RGB")
    plt.show()



# Colocar imagenes en una misma ventana una al lado de otra.
def concatenar_2imagenes(img1,img2,title='concatenado'):
#    if (img3.all() == 500):
#        numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)
#    elif(img4.all() == 500 and img3.all() != 500):
#        numpy_horizontal_concat = np.concatenate((img1, img2, img3), axis=1)
#    elif(img1.all() != 500 and img2.all() != 500 and img3.all() != 500 and img4.all() != 500):
#        numpy_horizontal_concat = np.concatenate((img1, img2, img3), axis=1)
#    else:
#        print('se necesitan por lo menos dos imágenes')
    numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)
    plt.figure()
    plt.axis("off")
    
    # En python el orden de los argumentos puede cambiarse si se nombra la variable
    # Se extablece un nivel mínimo y máximo para el mapa de colores, y se setea este en gray
#    graficar(numpy_horizontal_concat,vmax = 255, vmin = 0, cmap='gray')
    plt.imshow(numpy_horizontal_concat, vmax = 255, vmin = 0, cmap = 'gray')
    plt.title(title)



# =============================================================================
# graficar planos RGB para analizar
# =============================================================================
def analizarRGB(img, flag='histograma'):
    img=np.uint8(np.copy(img))
    #graficar la imagen separada en cada plano de color
    b,g,r = cv.split(img) #separar imagen en sus planos de color
    plt.figure()
    plt.subplot(221),plt.imshow(img[:,:,::-1]),plt.title('img original')
    plt.subplot(222),plt.imshow(b,'gray'),plt.title('BLUE')
    plt.subplot(223),plt.imshow(g,'gray'),plt.title('GREEN')
    plt.subplot(224),plt.imshow(r,'gray'),plt.title('RED')
    plt.show()
   
    #analizar el histograma en rgb
    if(flag=='histograma'):
        color = ('blue','green','red')
        plt.figure()
        for i,col in enumerate(color):
            histr = cv.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.title("histograma RGB")
        plt.show()
        
        
# =============================================================================
# graficar planos HSV para analizar
# =============================================================================  
def analizarHSV(img,flag='histograma'):
    img_patron = np.copy(img).astype(np.uint8)
    
    #pasar imagen a HSV
    img_patron = cv.cvtColor(img_patron, cv.COLOR_BGR2HSV) 
    H,S,V = cv.split(img_patron) #separar imagen en sus planos
    plt.figure()
    plt.subplot(131),plt.imshow(H,'gray'),plt.title('hue-tono')
    plt.subplot(132),plt.imshow(S,'gray'),plt.title('saturacion')
    plt.subplot(133),plt.imshow(V,'gray'),plt.title('value')
    plt.show()
    
    #analizar el histograma HSV
    if(flag=='histograma'):
        plt.figure()
        hist_h = cv.calcHist([img_patron],[0],None,[180],[0,180])
        hist_s = cv.calcHist([img_patron],[1],None,[256],[0,256])
        hist_v = cv.calcHist([img_patron],[2],None,[256],[0,256])
        plt.subplot(131),plt.plot(hist_h),plt.title('hue-tono')
        plt.subplot(132),plt.plot(hist_s),plt.title('saturacion')
        plt.subplot(133),plt.plot(hist_v),plt.title('value')
        plt.show()


# =============================================================================
# BGR a HSV
# =============================================================================
def BGRtoHSV(img_BGR):
    img_HSV = cv.cvtColor(img_BGR,cv.COLOR_BGR2HSV).astype(np.uint8)
    H,S,V = cv.split(img_HSV)
    
    return img_HSV,H,S,V


def BGRtoHSVR_merge(B,G,R):
    img_BGR = cv.merge((B,G,R))
    img_HSV = cv.cvtColor(img_BGR,cv.COLOR_BGR2HSV).astype(np.uint8)
    H,S,V = cv.split(img_HSV)
    
    return img_HSV,H,S,V
# =============================================================================
# HSV a BGR
# =============================================================================
def HSVtoBGR(img_HSV):
    imgBGR=cv.cvtColor(img_HSV,cv.COLOR_HSV2BGR).astype(np.uint8)
    B,G,R = cv.split(imgBGR)
    
    return imgBGR,B,G,R


def HSVtoBGR_merge(H,S,V):
    img_HSV = cv.merge((H,S,V))
    imgBGR=cv.cvtColor(img_HSV,cv.COLOR_HSV2BGR).astype(np.uint8)
    B,G,R = cv.split(imgBGR)
    
    return imgBGR,B,G,R



#     CIRCULOS COMO CONTORNO EN LA IMAGEN
def circle_contours(img, img_bin):
    #encontrar los contornos de una imagen binaria
    imagen_contornos, contornos, jerarquia = cv.findContours(img_bin,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    #encontrar y dibujar el círculo que cubre completamente el objeto con un área mínima
    for cnt in contornos:
        (x,y),radius = cv.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        img = cv.circle(img,center,radius,(255,255,255),2)


def redimensionar(img,h,w):
    alto,ancho = img.shape[:2]
    if(alto<h or ancho<w):
        res = cv.resize(img,(h, w), interpolation = cv.INTER_CUBIC)
    else:
        res = cv.resize(img,(h, w), interpolation = cv.INTER_LINEAR)
    
    return res



# =============================================================================
# #############################################################################
# #############################################################################
# #----------------------------- GENERACIÓN RUIDO -----------------------------
# #############################################################################
# #############################################################################
# =============================================================================
def ruidoGaussiano(img, mu, sigma):
    #calcular los parametros de la funcion
    H,W=img.shape

    #se genera el ruido
    ruido = np.random.normal(mu, sigma, size=[H,W])
        
    return img+ruido

def ruidoSalPimienta(img, s_vs_p, cantidad):
    # Parametros de entrada
    # img: imagen
    # s_vs_p: relacion de sal y pimienta (0 a 1)
    # cantidad: cantidad de ruido

    # Funcion para ensuciar una imagen con ruido sal y pimienta
    (alto, ancho) = img.shape
    # generar ruido tipo sal
    n_sal = np.ceil(cantidad * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(n_sal)) for i in img.shape]
    img[coords] = 255
    # generar ruido tipo pimienta
    n_pim = np.ceil(cantidad * img.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(n_pim)) for i in img.shape]
    img[coords] = 0
    return img
    
    
# =============================================================================
# =============================================================================
# #         OPEARCIONES ARITMÉTICAS
# =============================================================================
# =============================================================================

# =============================================================================
# suma de imagenes en un solo canal
# =============================================================================
#ojo que las imagenes deben ser del mismo tamaño
def suma(imgs):
    cant_imagenes = imgs.shape[0]
    print("cantidad de ims: ",cant_imagenes)
    for i in range(cant_imagenes):
        imgs[i]=np.float64(imgs[i])

    accum = np.copy(imgs[0])
    
    for i in range(1,cant_imagenes):
        accum = accum + imgs[i]
    
    accum = accum / cant_imagenes
    accum[accum>255]=255
    
    return accum

def interpolar(img1, img2, alpha):
    # SUMA imágenes
    # La suma es una mezcla lineal (alpha bledding) de 2 imágenes, es decir, una interpolación
    # b(x, y) = (1 − a)f(x, y) + ag(x, y), 0<=a<=1
    img3 = img1 * alpha + img2 * (1-alpha)
    return img3


# =============================================================================
# suma con peso entre 2 imagenes
# =============================================================================
#ojo que las imagenes deben ser del mismo tamaño
def suma_con_peso(img1,img2,p1,p2,constante=0):
    #la funcion de openCV solo funciona con imgs a color. Se agrega soporte a grises
    if(np.size(img1.shape)==2):
        img1=cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    if(np.size(img2.shape)==2):
        img2=cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
  
    #suma con pesos (blending). La constante se suma a cada valor una vez hecho el blending
    img_compuesta = cv.addWeighted(img1,p1,img2,p2,constante)
    return img_compuesta


def diferencia(img1, img2):
    img3 = img1 - img2

    # img3 -= img3.min()
    # img3 *= 255/img3.max()

    img3 += 255
    img3 = img3/2
    
#    img3 = img1/2-img2/2
#    img3 += 255
    
    return img3

def tr_potencial(img,gamma):
    print("transformacion potencial sobre valores de la imagen")
    img_modified = np.divide(img,256)
    img_modified = np.power(img_modified,gamma)
#    plt.imshow(img)
#    plt.show()
#    plt.imshow(img_modified)
#    plt.show()
    return img_modified


def operaciones_aritmeticas(img1, img2, tipo):
    TYPES = {
        "SUMA": cv.add(img1, img2),
        "RESTA": cv.subtract(img1, img2),
        "DIVISION": cv.divide(img1, img2),
        "MULTIPLICACION": cv.multiply(img1, img2),
    }

    return TYPES[tipo]


# =============================================================================
# BORRADO DE OBJETOS EN MOVIMIENTO con fondo estático
# =============================================================================
# OJO QUE imgs NO es una imagen si no un conjunto de imágenes (usar loadImages)
def borrado_obj_movimiento(imgs):
    accum_imgs = np.copy(imgs[0])
    cantidad_imagenes = imgs.shape[0]
    for i in range(1,cantidad_imagenes):  
        #cambiar espacio de color
        img_HSV = cv.cvtColor(imgs[i],cv.COLOR_BGR2HSV).astype(np.uint8)
        H,S,V = cv.split(img_HSV)
        
        # FILTRAR A CONVENIENCIA
#        aux1=np.array([V>225])
#        aux2=np.array([V<244])
#        V[aux1[0]==aux2[0]]=0
        
        #eliminar ruido sal del las nubes que quedo en la op anterior
        V = rst.filtro_mediana(V,3)
        V = np.uint8(V)
        
        #juntar los planos de la imagen nuevamente
        img_HSV = cv.merge((H,S,V))
        
        img_filtrada=cv.cvtColor(img_HSV,cv.COLOR_HSV2BGR).astype(np.uint8)
        
        #debe recordar mucho las fotos anteriores para mantener lo estatico (el estadio)
        #por lo tanto le doy un peso grande a la foto anterior
        accum_imgs = suma_con_peso(accum_imgs,img_filtrada,0.85,0.15)
    return accum_imgs



# =============================================================================
# #############################################################################
# #############################################################################
# #------------------------------MORFOLOGIA---------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# =============================================================================
# Engrosa objetos ya reconstruidos
# =============================================================================
def engrosamiento(img):
    elemEstruc = np.ones((3, 3), np.uint8)
    imgBordesL = img.copy()
    imgBordesL = imgBordesL.max() - imgBordesL
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            imgBordesL[i, j] = 0

    imgBordesAnterior = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # Se repite el paso del while, pero esta vez no respecto a la imágen binaria original si no a la de bordes
    # Esto causa que se rellenen un poco más los glóbulos
    while ((imgBordesL != imgBordesAnterior).max()): #es TRUE donde sean diferentes
        imgBordesAnterior = imgBordesL.copy()
        #Diltar
        dilatacion = cv.dilate(imgBordesAnterior, elemEstruc, iterations=1)

        #And
        imgBordesL = np.bitwise_and(dilatacion, (img.max() - img))
    
    imgFinal = imgBordesL.max() - imgBordesL
    
    return imgFinal

# =============================================================================
# Reconstruye objetos respecto a la imagen original RESPECTO AL BORDE DE LA PANT
# =============================================================================
def reconstruccion (img):
#     Los objetos de img deben ser binarizados y opuestos a la imagen original
    elemEstruc = np.ones((3, 3), np.uint8)

    # Se saca todo el relleno hasta dejar bordes muy finos 
    # (se reconstruye a partir de los bordes, editar a conveniencia)
    imgBordes = img.copy()
    for i in range(2, imgBordes.shape[0] - 2):
        for j in range(2, imgBordes.shape[1] - 2):
            imgBordes[i, j] = 0
    
    
    # Reconstruccion morfologica
    #Ciclo, se repite hasta que la img anterior sea igual a la actual
    imgBordesAnterior = np.zeros((imgBordes.shape[0], imgBordes.shape[1]), np.uint8)
    c = 0
    while ((imgBordes != imgBordesAnterior).max()): #es TRUE donde sean diferentes
        imgBordesAnterior = imgBordes.copy()
        #Diltar
        dilatacion = cv.dilate(imgBordesAnterior, elemEstruc, iterations=1)
        c+=1
        #And
        # SE RECONSTRUYE RESPECTO A LA IMAGEN ORIGINAL
    imgBordes = np.bitwise_and(dilatacion, img)
    
    return imgBordes


# =============================================================================
# operacion morfologica de apertura(erosion+dilatacion)
#    -trabaja sobre mascaras o contornos binarios
# =============================================================================
def apertura(img_binaria,kernel):
    img_resultado=np.copy(img_binaria).astype(np.uint8)
    img_resultado = cv.morphologyEx(img_resultado, cv.MORPH_OPEN, kernel)
    return img_resultado


# =============================================================================
# operacion morfologica de cierre(dilatacion+erosion)
# =============================================================================
def cierre(img_binaria,kernel):
    img_resultado=np.copy(img_binaria).astype(np.uint8)
    img_resultado = cv.morphologyEx(img_resultado, cv.MORPH_CLOSE, kernel)
    return img_resultado


# CIERRE: suaviza contornos, elimina agujeros pequeño, fusiona discontinuidades estrechas
# Elimina ruido interno del objeto
    # También se puede hacer con: opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
def cierre_p(img, kernel, it = 1):
    img_dilated = cv.dilate(img,kernel, iterations = it)
    # func.graficar(C, 255, 0, 'gray', 'Dilatacion')
    img_ap = cv.erode(img_dilated,kernel, iterations = it)

    return img_ap


# APERTURA: suaviza contornos, rompe puntas, elimina salientes delgadas
# Sirve para eliminar ruido que se encuentre fuera de la estructura.
# aumentar el tamaño del elemento estructurante para eliminar marcas de ruido mas grandes o cambiar el número de iteraciones
    # También se puede hacer con: opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
def apertura_p(img, kernel, it = 1):
    img_er = cv.erode(img,kernel, iterations = it)
    # func.graficar(C, 255, 0, 'gray', 'Erosion')
    img_c = cv.dilate(img_er,kernel, iterations = it)

    return img_c





# =============================================================================
# #############################################################################
# #############################################################################
# #---------------------- SEGMENTACION - CONTORN -MASK ------------------------
# #############################################################################
# #############################################################################
# =============================================================================

# GENERA KERNELS
def genQuickKernel(tam_kernel,tipo='rectangulo'):
    if(tipo=='rectangulo'):
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(tam_kernel,tam_kernel))
    if(tipo=='circulo'):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(tam_kernel,tam_kernel))
    if(tipo=='cruz'):
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,(tam_kernel,tam_kernel))
    return kernel


#KERNELS:
#     kernelcuadrado = np.ones((3,3),np.uint8)
#     kernelcruz = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
#     kernelelipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


# =============================================================================
# construir una mascara binaria de un objeto a partir de sus bordes
# =============================================================================
def mask_bordes(img_bordes):
    mask = np.copy(img_bordes).astype(np.uint8)
    
    #corregir los bordes de la imagen por si tienen huecos (no corrige ruido sal)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    #encontrar la mascara
    imagen_cnt, contornos, jerarquia = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    mask=cv.drawContours(mask,contornos,-1,255,-1)
    
    return mask


# =============================================================================
# rellena un sector de una imagen en gris, especificando valores minimo y maximo de
# gris, un pixel semilla para indicar desde donde comenzar el analisis de vecinos,+
# y el color con el que se va llenando. Es imprecindible que una region sea homogenea
# y este separada de otras regiones homogeneas con bordes bien definidos
# =============================================================================
def floodFill(img,lowest_value,gretest_value,point,colorval,mascara=None):
    src = img.copy()
    
    connectivity = 8 #8-adyacencia
    flags = connectivity #considerar vecinos
    flags |= cv.FLOODFILL_FIXED_RANGE
    
    #floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]]) -> retval, image, mask, rect
    cv.floodFill(src, mascara, point, colorval, lowest_value, gretest_value, flags)
    plt.figure()
    plt.imshow(src,'hot')
    plt.show()
    # =============================================================================

def bordes_roberts(img):
    img=np.float32(img)
    
    #se crean los kernels
    kernel_horizontal = np.array([
            [0,1],
            [-1,0]
            ])
    
    kernel_vertical = np.array([
            [1,0],
            [0,-1]
            ])

    #convolucionar ambos kernels y obtener los resultados
    img_horiz=cv.filter2D(img, -1, kernel_horizontal)
    img_vert=cv.filter2D(img, -1, kernel_vertical)
    
    #graficar para ver resultados parciales
    plt.figure()
    plt.subplot(121),plt.imshow(img_horiz,'gray'),plt.title('bordes horizontales')
    plt.subplot(122),plt.imshow(img_vert,'gray'),plt.title('bordes verticales')
    plt.show()
    
    #calcular la magnitud del gradiente roberts
    img_result = np.abs(img_horiz) + np.abs(img_vert)
    img_result =img_result /2
    
    img_result = np.uint8(img_result)
    
    #se binariaza la imagen    
    res,img_result=cv.threshold(img_result,25,256,cv.THRESH_BINARY)

    return img_result


#Cuando el tamaño del kernel es 3, el kernel de Sobel que se muestra arriba 
#puede producir imprecisiones notables (después de todo, Sobel es solo una 
#aproximación de la derivada).
def bordes_Sobel_cv(img):    
    #utilizar las funciones de openCV
    img_horiz = cv.Sobel(img,cv.CV_32F,1,0,ksize=3)
    img_vert = cv.Sobel(img,cv.CV_32F,0,1,ksize=3)
    
    #graficar para ver resultados parciales
    plt.figure()
    plt.subplot(121),plt.imshow(img_horiz,'gray'),plt.title('bordes Sobel horizontales')
    plt.subplot(122),plt.imshow(img_vert,'gray'),plt.title('bordes Sobel verticales')
    plt.show()
    
    #calcular la magnitud del gradiente 
#    img_result = cv.add(np.abs(img_horiz), np.abs(img_vert))
    img_result = np.abs(img_horiz) + np.abs(img_vert)
    img_result =img_result /2
    
    img_result = np.uint8(img_result)
    
    #se binariaza la imagen    
    res,img_result=cv.threshold(img_result,60,256,cv.THRESH_BINARY)

    return img_result


def bordes_sobel_prewitt(img,tipo):
    img=np.float32(img)
    
    #seleccion de detector
    if(tipo=="prewitt"): K=1
    if(tipo=="sobel"): K=2
    
    #se crean los kernels
    kernel_vertical = np.array([
            [-1,0,1],
            [-K,0,K],
            [-1,0,1],
            ])
    
    kernel_horizontal = np.array([
            [-1,-K,-1],
            [0,0,0],
            [1,K,1]
            ])
    
    kernel_diagonal_creciente = np.array([
            [0,1,K],
            [-1,0,1],
            [-K,-1,0]
            ])
    
    kernel_diagonal_decreciente = np.array([
            [-K,-1,0],
            [-1,0,1],
            [0,1,K]
            ])
    
    #convolucionar ambos kernels y obtener los resultados
    img_horiz = cv.filter2D(img, -1, kernel_horizontal) # el -1 es la profundidad (ddepth), si es -1, la profundidad es la misma que la imagen fuente
    img_vert = cv.filter2D(img, -1, kernel_vertical)
    img_diag_creciente = cv.filter2D(img, -1, kernel_diagonal_creciente)
    img_diag_decreciente = cv.filter2D(img, -1, kernel_diagonal_decreciente)
    
    #graficar para ver resultados parciales
    plt.figure()
    plt.subplot(221),plt.imshow(img_horiz,'gray'),plt.title(tipo + ' - bordes horizontales')
    plt.subplot(222),plt.imshow(img_vert,'gray'),plt.title(tipo + ' - bordes verticales')
    plt.subplot(223),plt.imshow(img_diag_creciente,'gray'),plt.title(tipo + ' - bordes diag creciente')
    plt.subplot(224),plt.imshow(img_diag_decreciente,'gray'),plt.title(tipo + ' - bordes diag decreciente')
    plt.show()
    
    #calcular la magnitud del gradiente
    img_result = np.copy(img_horiz)
    img_result = np.abs(img_result) + np.abs(img_vert)
    img_result = np.abs(img_result) + np.abs(img_diag_decreciente)
    img_result = np.abs(img_result) + np.abs(img_diag_creciente)
    img_result = img_result / 3

#    img_result = np.uint8(img_result)
        
    #se binariaza la imagen    
    res,img_result=cv.threshold(img_result,170,255,cv.THRESH_BINARY)

    return img_result




#OpenCV resuelve esta imprecisión para los kernels de tamaño 3 utilizando la función Scharr
def bordes_scharr(img):
    img=np.float32(img)
    
    #utilizar las funciones de openCV
    img_horiz = cv.Scharr(img,cv.CV_32F,1,0)
    img_vert = cv.Scharr(img,cv.CV_32F,0,1)
    
    #graficar para ver resultados parciales
    plt.figure()
    plt.subplot(121),plt.imshow(img_horiz,'gray'),plt.title('bordes scharr horizontales')
    plt.subplot(122),plt.imshow(img_vert,'gray'),plt.title('bordes scharr verticales')
    plt.show()
    
    #calcular la magnitud del gradiente
    img_result = np.copy(img_horiz)
    img_result = np.abs(img_result) + np.abs(img_vert)
    img_result = img_result / 2

    img_result = np.uint8(img_result)
    
    #se binariaza la imagen    
    res,img_result=cv.threshold(img_result,40,256,cv.THRESH_BINARY)

    return img_result


def bordes_laplaciano(img):
    #utilizar las funciones de openCV
    img_result = cv.Laplacian(img,cv.CV_8U)
    
    #graficar para ver resultados parciales
    plt.figure()
    plt.imshow(img_result,'gray'),plt.title('bordes laplaciano')
    plt.show()
        
    #se binariaza la imagen    
    res,img_result=cv.threshold(img_result,40,256,cv.THRESH_BINARY)

    return img_result


def bordes_LoG(img, umbral1, umbral2):
    kernel = np.array([
            [0, 0, -1, 0, 0],
              [0, -1, -2, -1, 0],
              [-1, -2, 16, -2, -1],
              [0, -1, -2, -1, 0],
              [0, 0, -1, 0, 0]])
    output_image = cv.filter2D(img, -1, kernel)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image



def bordes_canny(img):
    
    #utilizar la funcion de openCV
    img_result = cv.Canny(img,190,260)
    
    #graficar
    plt.figure()
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Imagen original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_result,cmap = 'gray')
    plt.title('Canny - Bordes de la Imagen'), plt.xticks([]), plt.yticks([])
    plt.show()

    return img_result

# =============================================================================
#     TRANSFORMADA DE HOUGH
# =============================================================================
def hough(mask,umbral):
    # mask deben ser bordes
    lines = cv.HoughLines(mask,1,np.pi/180,umbral)
    
    #hough ordena las lineas de mayor a menor tamaño.
    for rho,theta in lines[lines.shape[0]-1]:
        #encontrar algunos puntos de la linea para graficarla
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(mask,(x1,y1),(x2,y2),170,4)   
    return mask,rho,theta
    


# =============================================================================
#     ROTAR IMAGEN
# =============================================================================
def rotar_img(img,ang,hough='no hough'):
    # hough == 'hough' para cuando queremos rotar según el angulo de la transformada 
    rows,cols = img.shape
    if(hough == 'hough'):
        angulo = 180+np.rad2deg(ang)
    else:
        angulo = np.rad2deg(ang)
    M = cv.getRotationMatrix2D((cols/2,rows/2),angulo,1)
    img = cv.warpAffine(img,M,(cols,rows))
    
    return img


# =============================================================================
#     HOUGH CON TRACKBAR
# =============================================================================
def hough_tb(img):
    "Una línea se puede representar como y = mx + c o en forma paramétrica donde \rho es la distancia perpendicular desde el origen a la recta, y \theta es el ángulo formado entre esta perpendicular a la recta y el eje horizontal, medido en sentido contrario a las agujas del reloj." 
    "cv.HoughLines() Simplemente devuelve una matriz de valores (\rho, \theta), donde\rho se mide en píxeles como la distancia de la linea al centro de coordenadas, y \theta se mide en radianes representando la rotacion de dicha linea respecto al eje vertical. "
    "El primer parámetro, la imagen de entrada, debe ser una imagen binaria, por lo tanto, aplique el umbral o use la detección de bordes astutos antes de aplicar la transformada de Hough. Los parámetros segundo y tercero son las precisiones en \rho y  \theta respectivamente. El cuarto argumento es el umbral, es decir, el acumulado mínimo que debe obtener para que se le considere como una línea. Recuerde, el número de votos (acumulados) dependerá del número de puntos en la línea."
    
#    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = redimensionar(img,500,500)
    
    #detectar los bordes para luego utilizar hough
    edges = cv.Canny(img,30,180)
    
    
    #crear ventana para mostrar imagen e interactuar
    window_name = 'Hough'
    cv.namedWindow(window_name,flags = cv.WINDOW_AUTOSIZE | cv.WINDOW_FREERATIO | cv.WINDOW_GUI_EXPANDED)
    print("presione 'q' para terminar con el proceso")
    
    #crear barra de desplazamiento para cambiar parametro de acumulados
    title_trackbar1 = 'accum'
    max_accum = 150
    min_accum = 30
    cv.createTrackbar(title_trackbar1, window_name,min_accum,max_accum,nothing)
    cv.setTrackbarMin(title_trackbar1, window_name,min_accum)
    
    title_trackbar2 = 'line sizes'
    max_accum_l = 1000
    min_accum_l = 50
    cv.createTrackbar(title_trackbar2, window_name,min_accum_l,max_accum_l,nothing)
    cv.setTrackbarMin(title_trackbar2, window_name,min_accum_l)
    
    while(1):
        #copiar imagen original para no alterarla
        letra1_N = np.copy(img)
        
        # obtener los valores de la barra
        accum = cv.getTrackbarPos(title_trackbar1,window_name)
        
        #Obtiene el tamaño de las lineas desde la barra 
        size_l = cv.getTrackbarPos(title_trackbar2,window_name)
        
        #aplicar la transformada de Hough
        lines = cv.HoughLines(edges,1,np.pi/180,accum)    
        
        #recorrer las lineas
        for l in lines:
            for rho,theta in l:
                #encontrar algunos puntos de la linea para graficarla
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + size_l*(-b))
                y1 = int(y0 + size_l*(a))
                x2 = int(x0 - size_l*(-b))
                y2 = int(y0 - size_l*(a))
            cv.line(letra1_N,(x1,y1),(x2,y2),(0,0,255),2)
        
        #refrescar imagen con nuevos resultados
        cv.imshow(window_name, letra1_N)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cv.destroyAllWindows()


# =============================================================================
#     HOUGH PROBABILISTICO CON TRACKBAR
# =============================================================================
def houghP_tb(img):
    # =============================================================================
    # transformada probabilistica de Hough (HoughLinesP)
    # =============================================================================
    "La transformada probabilística de Hough es una optimización de la Transformada de Hough que vimos. No toma todos los puntos en consideración, en su lugar toma solo un subconjunto de puntos al azar y eso es suficiente para la detección de línea. Sólo se encesita disminuir el umbral."
    "cv2.HoughLinesP(), la cual posee dos nuevos argumentos:"
    "MinLineLength: longitud mínima de la línea. Los segmentos de línea más cortos que esto son rechazados. maxLineGap: espacio máximo permitido entre los segmentos de línea para tratarlos como una sola línea."
    
#    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = redimensionar(img,500,500)
    
    #detectar los bordes para luego utilizar hough
    edges = cv.Canny(img,70,150)
    
    #crear ventana para mostrar imagen e interactuar
    window_name = 'Hough Prob'
    cv.namedWindow(window_name,flags = cv.WINDOW_AUTOSIZE | cv.WINDOW_FREERATIO | cv.WINDOW_GUI_NORMAL)
    print("presione 'q' para terminar con el proceso")
    
    #crear barra de desplazamiento para cambiar parametros
    title_trackbar1 = 'accum:'
    title_trackbar2 = 'minLineLength:'
    title_trackbar3 = 'maxLineGap:'
    max_LineLength = 150
    max_LineGap = 30
    max_accum = 50
    cv.createTrackbar(title_trackbar1, window_name , 1, max_accum, nothing)
    cv.createTrackbar(title_trackbar2, window_name , 0, max_LineLength, nothing)
    cv.createTrackbar(title_trackbar3, window_name , 0, max_LineGap, nothing)
    cv.setTrackbarMin(title_trackbar1, window_name,1)
    cv.setTrackbarMin(title_trackbar2, window_name,0)
    cv.setTrackbarMin(title_trackbar3, window_name,0)
    
    while(1):
        #copiar imagen original para no alterarla
        letra1_P = np.copy(img)
        
        # obtener los valores de la barra
        accum = cv.getTrackbarPos(title_trackbar1,window_name)
        LineLength = cv.getTrackbarPos(title_trackbar2,window_name)
        LineGap = cv.getTrackbarPos(title_trackbar3,window_name)
        
        #aplicar la transformada de Hough probabilistica
        lines = cv.HoughLinesP(edges,1,np.pi/180,accum,LineLength,LineGap)
        
        #recorrer las lineas
        for l in lines:
            for x1,y1,x2,y2 in l:
                cv.line(letra1_P,(x1,y1),(x2,y2),(0,255,0),2)
        
        #refrescar imagen con nuevos resultados
        cv.imshow(window_name, letra1_P)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cv.destroyAllWindows()
    


##   CRECIMIENTO DE REGIONES
def floodFill_tb(img):
#    "el color que deseemos extraer no sea uniforme podemos establecer un rango de tolerancia, es decir un mínimo y máximo permito de cercanía"
#    "Primero la variable connectivity nos permite definir el nivel de conectividad a tomar en cuenta al analizar a los pixeles vecinos, puede ser 4 o 8, al establecer el flag cv2.FLOODFILL_FIXED_RANGE se tomará en cuenta la distancia entre los vecinos"
    def floodFill(tolerancia,point):
        src = img.copy()
        
        connectivity = 8 #8-adyacencia
        flags = connectivity #considerar vecinos
        flags |= cv.FLOODFILL_FIXED_RANGE
        
        lowest_value = (tolerancia,) * 3
        gretest_value = (tolerancia,) * 3
        
        #floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]]) -> retval, image, mask, rect
        cv.floodFill(src, None, point, (0, 255, 255), lowest_value, gretest_value, flags)
        cv.imshow('relleno', src)
    
    def mouse_clic(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            global point
            point = (x, y)
            tol = cv.getTrackbarPos(trackbar_name,winname)
            floodFill(tol,point)
    
    def trackbar_value(value):
        floodFill(value,point)
    
    
    #definir ventana interactiva
    winname = 'Flood fill'
    trackbar_name = 'Tolerancia'
    cv.namedWindow(winname)
    
    #definir elementos interactivos sobre ventana
    min_tolerancia = 1
    point = (0, 0)
    cv.setMouseCallback(winname, mouse_clic, img)
    cv.createTrackbar(trackbar_name, winname, min_tolerancia, 100, trackbar_value)
    cv.setTrackbarMin(trackbar_name, winname, min_tolerancia)
    
    while(1):
        cv.imshow(winname, img)
    #    if cv.waitKey(20) & 0xFF == 27:
    #        break
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
                break
    
    cv.destroyAllWindows()


def watershed(img,markers):
    # Descripción: coloca bordes de objetos
    # img: imagen original
    # markers: etiquetas 
    markers2 = cv.watershed(img,markers)

    img[markers2 == -1] = [255,0,0]
    
    return img


def labels(img):
    # La imagen debe ser binaria!
    # MARKERS: pintará los objetos con nº de 1 a Max objetos. Es decir, si se tienen 5 objetos
    # El primer objeto tendrá todos valores 1, el segundo todos valores 2, etc. Cada objeto está rodeado por -1
    # RET: contendrá la cantidad de objetos + el fondo
    ret, markers = cv.connectedComponents(img)

    # CANTIDAD DE PIXELES (AREA) DE OBJETOS --> detectar mayor o cosas segun su tamaño
#    s = (ret,1)
    area_obj = np.zeros(ret)
    for i in range(0,ret):
        area_obj[i] = np.count_nonzero(markers==i)

    #DEVUELVE EL índice (objeto) MÁS GRANDE:
#    print(np.argmax(area_obj))

    #Markers va a tener una etiqueta por cada rosa, por lo que si se cuenta se tiene la cantidad
#    cantidad_objetos = np.max(markers-1)
    cantidad_objetos = ret-1 # Lo mismo
    
    return cantidad_objetos, markers, area_obj
