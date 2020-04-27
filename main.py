import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
import funciones as func
#import restauracion as rst
import funciones_emi as funcEmi
import json
import medpy.metric as mdm


# =============================================================================
# COMPARACIÓN CON METADATOS DE JSON
# =============================================================================

# Lee los metadatos JSON para ver si es maligno o benigno
def benigno_maligno(address):
    with open(address) as file:
        data = json.load(file)
#        print('Benigno o maligno:', data['meta']['clinical']['benign_malignant'])
#        print('')
        return data['meta']['clinical']['benign_malignant']

# NOTA: los nombres de las imágenes no deben tener espacios
#       UTILIZAR LA CARPETA banco_imgs_json para comparación
#       Descomentar if del final
        
# Ejemplos
# =============================================================================
#     BENIGNOS
# =============================================================================

#img = cv.imread('banco_imgs_json/ISIC_0000005.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000005.json')

#img = cv.imread('banco_imgs_json/ISIC_0000010.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000010.json')

#img = cv.imread('banco_imgs_json/ISIC_0000019.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000019.json')

#img = cv.imread('banco_imgs_json/ISIC_0000032.jpg') # incorrecta # WRONG (muy parecido a un maligno)
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000032.json')

#img = cv.imread('banco_imgs_json/ISIC_0000079.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000079.json')


#
# =============================================================================
#   MALIGNOS
# =============================================================================

img = cv.imread('banco_imgs_json/ISIC_0000022.jpg')
tipo = benigno_maligno('banco_imgs_json/ISIC_0000022.json')

#img = cv.imread('banco_imgs_json/ISIC_0000026.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000026.json')

#img = cv.imread('banco_imgs_json/ISIC_0000029.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000029.json')

#img = cv.imread('banco_imgs_json/ISIC_0000030.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000030.json')

#img = cv.imread('banco_imgs_json/ISIC_0000031.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000031.json')

#img = cv.imread('banco_imgs_json/ISIC_0000076.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000076.json')

#img = cv.imread('banco_imgs_json/ISIC_0000077.jpg') 
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000077.json')

#img = cv.imread('banco_imgs_json/ISIC_0000078.jpg') # incorrecta
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000078.json')

#img = cv.imread('banco_imgs_json/ISIC_0000002.jpg')
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000002.json')

#img = cv.imread('banco_imgs_json/ISIC_0000031.jpg')  # DOS LUNARES
#tipo = benigno_maligno('banco_imgs_json/ISIC_0000031.json')

# =============================================================================
# #############################################################################
# #############################################################################
# -------------------- Redimensionado de imágenes -----------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# Algunas imagenes vienen con bordes blancos, por lo que se eliminaran 5px en cada
# uno de los lados para evitar esto. Se asume que mas o menos la mayoria de los
# lunares se ubican en el centro de la imagen por lo que no habra problema
img = img[6:img.shape[0]-5,6:img.shape[1]-5,:]

# redimensionar poco mas de la mitad de su tamaño
img = cv.resize(img,None,fx=0.6, fy=0.6, interpolation = cv.INTER_CUBIC)

# =============================================================================
# #############################################################################
# #############################################################################
# ------------------- Eliminación de esquina de lente--------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
#para que las esquinas de la imagen donde se ve la forma de la lente no influya en 
#los calculos, se crea una mascara circular de radio igual a la coordenada mas
#grande (alto o ancho). Esta mascara se utilizara para segmentar luego el lunar
h,w,c = img.shape
maskFoco=np.zeros((h,w))
if(h>w):
    cv.circle(maskFoco,(int(w/2),int(h/2)), int(h/2)-5, 255, -1)
else:
    cv.circle(maskFoco,(int(w/2),int(h/2)), int(w/2)-5, 255, -1)
maskFoco=np.uint8(maskFoco)
#func.graficar(maskFoco,'mascara circular de lente camara')



# =============================================================================
# #############################################################################
# #############################################################################
# ------------------------------ SEGMENTACIÓN ---------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#

# Se convierte la imagen a escala de grises
img_g=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
func.graficar(img_g,'imagen original en gris')

# Filtro de medias no locales
img_g= cv.fastNlMeansDenoising(img_g,None)

# Observación de histograma por si hay datos interesantes
func.histograma(img_g,grafico='true',titulo='histograma gray del lunar')

# Se obtiene máscara óptima
ret, img_mod_tresh = cv.threshold(img_g, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
print("umbral optimo de otsu",ret)

# Se mejora la máscara con morfologia
kernelelipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
img_mod = funcEmi.apertura(img_mod_tresh,kernelelipse)
kernelelipse = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
img_mod = funcEmi.apertura(img_mod,kernelelipse)

# Filtro y recuperacion de borde perdido
kernelelipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
img_mod = funcEmi.cierre(img_mod,kernelelipse)
img_mod = cv.dilate(img_mod,kernelelipse,1)



# Quitado de bordes de lente
img_mod= img_mod*maskFoco
img_mod_tresh = img_mod_tresh * maskFoco

funcEmi.graficar_plt2(img_mod_tresh,img_mod, 'Threshold', 'Morfología',16)

# =============================================================================
#   EXTRACCIÓN DE MELANOMA
# =============================================================================
# Se etiquetan los objetos
cantidad_objetos, markers, area_obj = funcEmi.labels(img_mod)

# Se copia para ordenar y obtener el valor más grande
area_obj2 = area_obj.copy()
area_obj2 = np.sort(area_obj2)
end = np.int(area_obj2.shape[0]) # Último valor (el más grande)

# Se obtiene el índice del objeto más grande (el melanoma)
area_obj=list(area_obj)
area_obj2=list(area_obj2)

# -2 si es oscuro -1 si es blanco!
elemento_mel = area_obj.index(area_obj2[end-2])

markers[markers == elemento_mel]=255
markers[markers < 255]=0

# Se extrae el melanoma de la imagen original
markers = np.uint8(markers)
melanoma_extraido_B = np.bitwise_and(img[:,:,0],markers)
melanoma_extraido_G = np.bitwise_and(img[:,:,1],markers)
melanoma_extraido_R = np.bitwise_and(img[:,:,2],markers)

melanoma_extraido = cv.merge((melanoma_extraido_B,melanoma_extraido_G,melanoma_extraido_R))



# =============================================================================
# #############################################################################
# #############################################################################
# -------------------------------- CONTORNOS ----------------------------------
# #############################################################################
# #############################################################################
# =============================================================================

# =============================================================================
#   CONVEX HULL y GRADIENTE
# =============================================================================
#utilizar funcion de la tesis para mejorar convex-hull(PAPER)
def ConvexHull(mascara):
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv.findContours(mascara1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv.convexHull(cnt)
    puntosConvex=hull[:,0,:]
    m,n=mascara1.shape
    ar=np.zeros((m,n))
    mascaraConvex=cv.fillConvexPoly(ar, puntosConvex, 255)
    mascaraConvex = np.uint8(mascaraConvex)
    imC, contoursC, hierarchyC = cv.findContours(mascaraConvex,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE )
    cntHULL = contoursC[0]
    return cntHULL

#calcular el contorno del convex hull
contornoHull = ConvexHull(markers)
cHull = np.zeros((markers.shape[0],markers.shape[1],3))
cv.drawContours(cHull, contornoHull, -1, (0,255,0), 2)
im2, contours, hierarchy = cv.findContours(markers,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(cHull, contours, -1, (255,0,0), 2)


# Para medpy
####################################################################
cHull_justHull = np.zeros((markers.shape[0],markers.shape[1],3))
cHull_justMelanoma = np.zeros((markers.shape[0],markers.shape[1],3))
cv.drawContours(cHull_justHull, contornoHull, -1, (0,255,0), 2)
cv.drawContours(cHull_justMelanoma, contours, -1, (255,0,0), 2)
####################################################################


funcEmi.graficar_plt4(img_mod,markers,img[:,:,::-1],melanoma_extraido[:,:,::-1],'Melanoma Umbralizado','Melanoma Umbralizado extraído','Imagen original', 'Melanoma Extraído',4)
#funcEmi.graficar_plt2(,5)
func.graficar(cHull,'ConvexHull',modo='color')
#func.graficar(melanoma_extraido,'Melanoma extraído',modo='color')


# =============================================================================
# #############################################################################
# #############################################################################
# ------------------------------ ASIMETRIAS -----------------------------------
# #############################################################################
# #############################################################################
# =============================================================================
#
# Se encuentran los contornos
imagen, contornos, jerarquia = cv.findContours(markers,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
print("se encontraron:",np.array(contornos).shape[0],"contornos")
cnt=contornos[0]

# Se calcula el momento
M = cv.moments(markers)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print("centroide del lunar",(cx,cy))
aux = melanoma_extraido.copy()
cv.rectangle(aux,(cx-10,cy-10),(cx+10,cy+10),(0,255,255),2)
#func.graficar(aux,'centroide del lunar',modo='color')

#area
area = cv.contourArea(cnt)
print("area del lunar",area)

#relacion de aspecto
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
print("relacion de aspecto",aspect_ratio)


#extension: razón entre el área del contorno y el área del BB circular delimitador.
(x,y),radius = cv.minEnclosingCircle(cnt)
area_circuloBB = np.pi * np.power(radius,2)
extension = float(area)/area_circuloBB
print("la extension del lunar es: ",extension)


#La solidez es la razón entre el área del contorno y el área de su envoltura convexa.
hull_area = cv.contourArea(contornoHull)
solidez = float(area)/hull_area
print("la solidez es",solidez)




# =============================================================================
# #############################################################################
# #############################################################################
# #-----------------------------COLORES-------------------------------
# #############################################################################
# #############################################################################
# =============================================================================


print("==========ANALISIS DE COLORES===============")
# Se analiza el color general del melanoma (si es muy oscuro -> mal indicio)
aux = np.copy(img)
aux= cv.fastNlMeansDenoisingColored(aux,None,5,5,7,15)
img_lunar_L = cv.cvtColor(img,cv.COLOR_BGR2LUV)

# Se utiliza el canal de luminancia que funciona mejor que GRAY
L,_,_ = cv.split(img_lunar_L)

# luminancia media del lunar (comparado con lunares oscuros feos)
mediaL = cv.mean(L,mask=markers)
print("la claridad general del lunar es: ",mediaL[0])

#MEDIA Y DESVIO DE CADA PLANO DE COLOR
#func.analizarRGB(aux,mask=markers)
#func.analizarHSV(aux,mask=markers)
mediaB,desvioB = cv.meanStdDev(aux[:,:,0],mask=markers)
mediaG,desvioG = cv.meanStdDev(aux[:,:,1],mask=markers)
mediaR,desvioR = cv.meanStdDev(aux[:,:,2],mask=markers)

# Se estudia el nivel de rojo del lunar comparando con el color de la piel
print("nivel de rojo global del lunar: ",mediaR[0])
mask_aux=255+(-1)*markers
mask_aux=np.uint8(mask_aux)
media_piel,desvio_piel = cv.meanStdDev(aux[:,:,2],mask=mask_aux)
print("nivel de rojo global de la piel es: ",media_piel[0])
dif_rojo_pielLunar = np.abs(media_piel[0]-mediaR[0])
print("diferencia rojo con piel: ",dif_rojo_pielLunar)

if(mediaL[0]<70):
    print("LUNAR MUY OSCURO")
if(dif_rojo_pielLunar>100):
    print("LUNAR MUY ROJIZO")

print("===========================")


# =============================================================================
# comprobar variaciones de color con estandares
# =============================================================================

# Se extrae solo el lunar sin el fondo (piel)
_, contornos, _ = cv.findContours(markers,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv.boundingRect(contornos[0])
lunar_segmentado = melanoma_extraido[y:y+h,x:x+w] #lunar centrado en Bounding Box
mask_segmentada = markers[y:y+h,x:x+w] #mascara lunar centrado en Bounding Box

# Se definen los 6 colores malos
bad_blanco = [255,255,255]
bad_negro = [0,0,0]
bad_rojo = [int(0.2*255),int(0.2*255),int(0.8*255)]
bad_cafeClaro= [0,int(0.4*255),int(0.6*255)]
bad_cafeOscuro= [0,0,int(0.2*255)]
bad_grisAzul= [int(0.6*255),int(0.6*255),int(0.2*255)]
bad_colores=np.array([bad_blanco,bad_negro,bad_rojo,bad_cafeClaro,bad_cafeOscuro,bad_grisAzul])
bad_colores_nombres=np.array(["blanco","negro","rojo","cafeClaro","cafeOscuro","grisAzul"])


# Se suaviza la imagen con un filtro gaussiano local
lunar_segmentado= cv.fastNlMeansDenoisingColored(lunar_segmentado,None,5,5,7,15)

# Se obtiene la cantidad de pixeles que representa el fondo y el lunar
px_lunar=cv.countNonZero(mask_segmentada)
px_fondo = (lunar_segmentado.shape[0]*lunar_segmentado.shape[1])-px_lunar

# Se calcula distancia euclidea de cada pixel respecto a los colores malos
contador_coloresMalos = 0
for i in range(6):
    lunar_segmentado_gray=cv.cvtColor(lunar_segmentado, cv.COLOR_BGR2GRAY)
    bad_color_gray=(bad_colores[i,0]*0.114)+(bad_colores[i,1]*0.587)+(bad_colores[i,2]*0.299)
    
    res=np.abs(np.float32(lunar_segmentado_gray)-bad_color_gray)
    res=cv.bitwise_and(res,res,mask=mask_segmentada)
    func.graficar(res,'distancia euclidea con color %s' %(bad_colores_nombres[i]))
    
    media,desvio = cv.meanStdDev(res,mask=mask_segmentada)
    print("media y desvio general respecto al color ",bad_colores_nombres[i],bad_color_gray)
    print(media[0],desvio[0])
    
    cant_pixeles = np.size(np.where(res<desvio)[0])
    cant_pixeles = cant_pixeles-px_fondo
    print("el porcentaje de color maligno actual del estandar es: ",(cant_pixeles/px_lunar))
    if((cant_pixeles/px_lunar)>0.2):#<<<------umbral % de pixeles / area total para considerar malo
        contador_coloresMalos=contador_coloresMalos+1
    print("===========================")

print("======>RESULTADO")
print("contador de indicador de colores malos: ", contador_coloresMalos)



# =============================================================================
#       HSV
# =============================================================================
img_HSV, H,S,V = funcEmi.BGRtoHSV(melanoma_extraido)

# Media de H
meanH = np.mean(H[H>0])
medianH = np.median([H>0])
stdH = np.std(H[H>0])


print(' ')
print('Valores estadísticos H')
print('Media H',meanH)

print('STD H',stdH)


# Se crea la máscara H
H2 = H.copy()
H2[H>0] = 255


funcEmi.graficar_plt3(H,S,V,'H','S','V',13)

funcEmi.graficar_plt2(img[:,:,::-1],H,'Imagen original','Hue',17)


# =============================================================================
# #############################################################################
# #############################################################################
# #--------------------- MEDPY distancia entre contornos ----------------------
# #############################################################################
# #############################################################################
# =============================================================================

# FUENTE: https://loli.github.io/medpy/metric.html


# Se obtienen imágenes binarias de los las imágenes donde se graficaron contornos
ret, cHull_binary = cv.threshold(cHull_justHull[:,:,1], 0, 255, cv.THRESH_BINARY)
ret, cHull_Mbinary = cv.threshold(cHull_justMelanoma[:,:,0], 0, 255, cv.THRESH_BINARY)


print(' ')

# =============================================================================
#     Distancia Hausdorff
# =============================================================================
distanceHD1 = mdm.hd(cHull_binary,cHull_Mbinary,connectivity=1)

print('Distancia Hausdorff',distanceHD1)



# =============================================================================
#   Distancia media entre superficies
# =============================================================================

distanceASD = mdm.asd(cHull_binary,cHull_Mbinary,connectivity=1)

print('Distancia Media',distanceASD)



# =============================================================================
#     Distancia media simétrica entre superficies
# =============================================================================
distanceASSD = mdm.assd(cHull_binary,cHull_Mbinary,connectivity=1)

print('Distancia Media Simétrica',distanceASSD)


# =============================================================================
#     Distancia media entre superficie de objetos
# =============================================================================

distanceObjASD = mdm.obj_asd(cHull_binary,cHull_Mbinary,connectivity=1)

print('Distancia Media entre superficie de objetos: ',distanceObjASD)



# =============================================================================
#     Distancia media simétrica entre superficie de objetos
# =============================================================================

distanceObjASSD = mdm.obj_assd(cHull_binary,cHull_Mbinary,connectivity=1)

print('Distancia Media Simétrica entre superficie de objetos: ',distanceObjASSD)



# =============================================================================
# #############################################################################
# #############################################################################
# #------------------------------ CLASIFICACIÓN -------------------------------
# #############################################################################
# #############################################################################
# =============================================================================


print('')

# NOTA: La mejor clasificación hasta el momento es la distancia media de la 
# envoltura convexa a los bordes, por lo que se priorizó esta.
# En cojunto con la misma se pone la varianza de color, la cual en combinación
# refuerza la clasificación. 
# Por otro lado la simetría entre la envoltura y los bordes 
if  distanceASD>4.72 and stdH>35 or distanceObjASSD>12 :
    clase = 'malignant'
    print('Clasificación del software: ', clase)
else:
    clase = 'benign'
    print('Clasificación del sofware: ', clase)
    
if tipo == clase:
    print('Clasificación dermatológica: ', tipo )
    print('Resultado de clasificación: correcta')
else:
    print('Clasificación dermatológica: ', tipo )
    print('Resultado de clasificación: incorrecta')
    