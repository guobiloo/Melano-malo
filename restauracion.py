# =============================================================================
# #--------------------------IMPORTS---------------------------------
# =============================================================================
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pdifunFixed as pdi


# =============================================================================
# filtros pasa-alto suma 1
#   - da un efecto de enfoque de imagen, realzando un poco los bordes difusos
# =============================================================================
def filtro_suma1(img,P=1):
    img_modified = np.copy(img)
    
    #crear kernel
    kernel = np.array(
                [[-1, -1, -1],
                 [-1,  9, -1],
                 [-1, -1, -1]])
    if(P<1):
        print("se mantiene el mismo kernel para evitar errores:",kernel)    
    if(P>1):
        K=P*8
        kernel = (-P)*np.ones((3,3))
        kernel[1,1]=K+1
    
    #aplicar filtro
    img_filtrada = cv.filter2D(img_modified,-1,kernel)
    return img_filtrada


# =============================================================================
# filtros pasa-alto suma 0
#   - realza mucho los bordes y elimina variaciones suaves (elimina el fondo)
# =============================================================================
def filtro_suma0(img,P=1):
    img_modified = np.copy(img)
    
    #crear kernel
    kernel = np.array(
                [[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]])
    if(P<1):
        print("se mantiene el mismo kernel para evitar errores:",kernel)    
    if(P>1):
        K=P*8
        kernel = (-P)*np.ones((3,3))
        kernel[1,1]=K
    
    #aplicar filtro
    img_filtrada = cv.filter2D(img_modified,-1,kernel)
    return img_filtrada


# =============================================================================
# filtros de alta potencia
#   - permite enfatizar altas frecuencias (enfoca bordes) y mantiene el fondo
#     pero de forma difuminada
# =============================================================================
def filtro_AltaPotencia(img, tam_kernel,A):
    img_modified = np.copy(img)
    
    #aplicar filtro pasa bajo para suavizar
    img_filtrada = filtro_uniforme(img_modified,tam_kernel)
    
    #amplifico los valores de la imagen
    img_aplificada = cv.multiply(img_modified,A)

    #hago la resta entre la imagen original y su version filtrada-suavizada
    img_modified = cv.subtract(img_aplificada,img_filtrada)
    
#    #reintegrar al rango de grises
#    img_modified[img_modified>255]=255
#    img_modified[img_modified<0]=0
    return np.uint8(img_modified)


# =============================================================================
# filtro pasa bajos uniforme
#  - Bueno para ruido gaussiano, falla con ruido impulsivo.
# =============================================================================
def filtro_uniforme(img, tam_kernel):
    img_modified=np.copy(img)
    img_modified=cv.blur(img_modified,(tam_kernel,tam_kernel))
    return img_modified


# =============================================================================
# filtro pasa bajos gaussiano
#  - Bueno para ruido gaussiano, falla con ruido impulsivo.
# =============================================================================
def filtro_gaussiano_PB(img, tam_kernel, mu, sigmaX=0, sigmaY=0):
    img_modified=np.copy(img)
    img_modified=cv.GaussianBlur(img_modified,(tam_kernel,tam_kernel), mu,sigmaX,sigmaY)
    return img_modified


# =============================================================================
# filtro media geometrica (Menos perdida de detalles que los anteriores)
#   - Bueno para ruido gaussiano, falla con ruido impulsivo.
# =============================================================================
def filtro_MediaGeometrica(img, tam_kernel):
    img_modified=np.copy(img).astype(np.float32)
    
    (s, t) = img_modified.shape
    for i in range(0, s-tam_kernel+1):
        for j in range(0, t-tam_kernel+1):
            acum = 1
            for k in range(i, i+tam_kernel):
                for o in range(j, j+tam_kernel):
                    acum = acum * img_modified[k, o]
            img_modified[i,j] = float(pow(acum, 1.0/(tam_kernel*tam_kernel)))
    return img_modified


# =============================================================================
# filtro media armonica
#   - Bueno para ruido sal, malo para ruido pimienta.
#   - Bien con ruido gaussiano, preservando detalles de la imagen.
# =============================================================================
def filtro_MediaArmonica(img, tam_kernel):
    img_modified=np.copy(img).astype(np.float32)
    
    (s, t) = img_modified.shape
    for i in range(0, s-tam_kernel+1):
        for j in range(0, t-tam_kernel+1):
            acum = 0
            for k in range(i, i+tam_kernel):
                for o in range(j, j+tam_kernel):
                    acum = acum + np.float((1/img_modified[k, o]))
            img_modified[i,j] = np.divide((tam_kernel*tam_kernel),acum)
    return img_modified


# =============================================================================
# filtro media contra armonica (engloba cualidades de mediaArmonica)
#   - Q: orden
#   - Q=0 (media aritmetica)
#   - Q=-1 (media armonica)
#   - (Q>0 elimina pimienta) // (Q<0 elimina sal)
# =============================================================================
def filtroMediaContraarmonica(img, Q, tam_kernel):
    (m, n) = img.shape
    img = img.astype(np.float32)
    for i in range(0, m-tam_kernel+1):
        for j in range(0, n-tam_kernel+1):
            cont1 = 0
            cont2 = 0
            for k in range(i, i+tam_kernel):
                for o in range(j, j+tam_kernel):
#                    cont1 = cont1 + np.power(img[k, o], Q+1)
#                    cont2 = cont2 + np.power(img[k, o], Q)
                    cont1 = cont1 + (img[k, o]** Q+1)
                    cont2 = cont2 + (img[k, o]** Q)
            img[i, j] = cont1 / cont2
    return img


# =============================================================================
# filtro de mediana
#    - Bueno para todo, pero el mejor para ruido impulsivo
# =============================================================================
def filtro_mediana(img,tam_kernel):
    img = img.astype(np.float32)
    
    #calcular el filtro de mediana
    return cv.medianBlur(img,tam_kernel)


# =============================================================================
# filtro de punto medio
#    - Util para ruido tipo gaussiano o uniforme
# =============================================================================
def filtro_puntoMedio(img,tam_kernel):
    img_modified = np.copy(img).astype(np.float)
    p_medio = np.int(np.ceil(tam_kernel/2))
    H,W=img.shape
    
    for i in range(0,H-tam_kernel+1):
        for j in range(0,W-tam_kernel+1):
            sub_img=img[i:i+tam_kernel,j:j+tam_kernel]
            maximo = np.max(sub_img)
            minimo = np.min(sub_img)
            img_modified[i+p_medio,j+p_medio] = 0.5*(minimo+maximo)
    return img_modified


# =============================================================================
# filtro de minimo
#    - util para ruido sal unicamente
# =============================================================================
def filtro_minimo(img,tam_kernel):
    H,W = img.shape
    img_modified = np.copy(img).astype(np.float)
    p_medio = np.int(np.ceil(tam_kernel/2))
    
    for i in range(0,H-tam_kernel+1):
        for j in range(0,W-tam_kernel+1):
            sub_img=img[i:i+tam_kernel,j:j+tam_kernel]
            minimo = np.min(sub_img)
            img_modified[i+p_medio,j+p_medio] = minimo
    return img_modified


# =============================================================================
# filtro de maxima
#    - util para ruido pimienta unicamente
# =============================================================================
def filtro_maximo(img,tam_kernel):
    H,W = img.shape
    img_modified = np.copy(img).astype(np.float)
    p_medio = np.int(np.ceil(tam_kernel/2))
    
    for i in range(0,H-tam_kernel+1):
        for j in range(0,W-tam_kernel+1):
            sub_img=img[i:i+tam_kernel,j:j+tam_kernel]
            maximo = np.max(sub_img)
            img_modified[i+p_medio,j+p_medio] = maximo
    return img_modified


# =============================================================================
# filtro adaptativo
#    - considerar siempre que varRuido<=varLocal
#    - si varRuido=0, se debe obtener imagen original
#    - si varLocal ~~ varRuido, se debe obtener la media
#    - si varLocal >>> varRuido, se obtiene valor cercano a la imagen
#    - solo utilizar cuando se puede estimar varRuido
#    - si varRuido es alto, fijar en 1 o utilizar filtro de media que es lo mismo
# =============================================================================
def filtro_adaptativo(source, varRuido, size):
    final = source.copy().astype(np.uint8)
    cols, rows = source.shape
    for y in range(size/2, rows - size/2):
        for x in range(size/2, cols - size/2):
            #tomar una porcion cuadrada de la imagen de size*size
            x1 = max(0, x-size/2)
            x2 = min(cols+1, x+size/2)
            y1 = max(0, y-size/2)
            y2 = min(rows+1, y+size/2)

            #calcular media y varianza local de ese cuadrado
            media = np.mean(source[x1:x2,y1:y2])
            varLocal = np.std(source[x1:x2,y1:y2])

            if np.isnan(varLocal):
                continue
            if int(varLocal) is not 0:
                aux = (source[x,y] - (varRuido/varLocal) * (source[x,y]-media))
                aux = min(aux, 255)
                aux = max(aux, 0)
                final[x, y] = int(aux)

    final.astype(np.uint8)
    return final

# =============================================================================
# ESPECTRO - filtro rechaza banda (para pasa banda, el hacer 1-esteFiltro)
# =============================================================================
def filtroRechazaBanda_butterworth(rows, cols, minimo,maximo, order):
    # Filtro pasa banda implementado con dos butterworth
    filtro_superior = pdi.filterButterworth(rows, cols, maximo, order)
    filtro_inferior = pdi.filterButterworth(rows, cols, minimo, order)
    filtro_superior = 1-filtro_superior
    filtroRechazaBanda = np.add(filtro_inferior,filtro_superior)
    return filtroRechazaBanda


# =============================================================================
# ESPECTRO - filtro notch ad-hoc (hacer cero alguna zona del espectro manualmente)
# =============================================================================
def filtroNotch(img, point, pixel = False):
    # Filtro notch implementado que aplica un pasa altos butterworth en el punto
    # Si pixel es true, en vez de butterworth, anula solo un circulo (tipo ideal)
    
    rows,cols=img.shape
    
    if not pixel:
        #Variables del filtro
        corte = 0.02
        order = 8

        #Creo el filtro Butterworth pasa bajos en frecuencia
        filtro = pdi.filterButterworth(rows, cols, corte, order)
        filtroPasaAlto = 1 - filtro
        filtroPasaAlto = np.roll(filtroPasaAlto, point[1], axis=0)
        filtroPasaAlto = np.roll(filtroPasaAlto, point[0], axis=1)
    else:
        filtroPasaAlto = np.ones([cols, rows])
        radius = 15
        filtroPasaAlto = cv.circle(filtroPasaAlto, (point[0], point[1]), radius, 0, -1)

    return filtroPasaAlto