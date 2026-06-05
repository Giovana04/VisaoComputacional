from calculo_volume import calculo_volume
from isosuperficie import isosuperficie
from esqueleto import esqueleto
volume = calculo_volume('trab3\\b0207')

superficie, volume = isosuperficie(volume)

esq = esqueleto(superficie, volume)