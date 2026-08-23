"""Koppen-Geiger climate lookup for SFR vegetation asset selection.

Generated from Beck et al. (2018) present-day Koppen-Geiger map:
- Beck_KG_V1_present_0p5.tif, WGS84, 0.5 degree grid
- Values map to legend.txt in the same dataset

The original dataset is CC BY 4.0. This compact module stores only the
0.5 degree class-id grid needed for tile-level forest asset selection.
"""

import base64
import functools
import zlib

GRID_WIDTH = 720
GRID_HEIGHT = 360
GRID_RES_DEG = 0.5
GRID_LON_W = -180.0
GRID_LAT_N = 90.0

KOPPEN_CLASSES = {1: ('Af', 'Tropical, rainforest'), 2: ('Am', 'Tropical, monsoon'), 3: ('Aw', 'Tropical, savannah'), 4: ('BWh', 'Arid, desert, hot'), 5: ('BWk', 'Arid, desert, cold'), 6: ('BSh', 'Arid, steppe, hot'), 7: ('BSk', 'Arid, steppe, cold'), 8: ('Csa', 'Temperate, dry summer, hot summer'), 9: ('Csb', 'Temperate, dry summer, warm summer'), 10: ('Csc', 'Temperate, dry summer, cold summer'), 11: ('Cwa', 'Temperate, dry winter, hot summer'), 12: ('Cwb', 'Temperate, dry winter, warm summer'), 13: ('Cwc', 'Temperate, dry winter, cold summer'), 14: ('Cfa', 'Temperate, no dry season, hot summer'), 15: ('Cfb', 'Temperate, no dry season, warm summer'), 16: ('Cfc', 'Temperate, no dry season, cold summer'), 17: ('Dsa', 'Cold, dry summer, hot summer'), 18: ('Dsb', 'Cold, dry summer, warm summer'), 19: ('Dsc', 'Cold, dry summer, cold summer'), 20: ('Dsd', 'Cold, dry summer, very cold winter'), 21: ('Dwa', 'Cold, dry winter, hot summer'), 22: ('Dwb', 'Cold, dry winter, warm summer'), 23: ('Dwc', 'Cold, dry winter, cold summer'), 24: ('Dwd', 'Cold, dry winter, very cold winter'), 25: ('Dfa', 'Cold, no dry season, hot summer'), 26: ('Dfb', 'Cold, no dry season, warm summer'), 27: ('Dfc', 'Cold, no dry season, cold summer'), 28: ('Dfd', 'Cold, no dry season, very cold winter'), 29: ('ET', 'Polar, tundra'), 30: ('EF', 'Polar, frost')}

_GRID_ZLIB_B64 = (
    "eNrtnYmWorwWhQMo6JIl3eWtskt+fP/HvJnnhIQZzOmhLEdMPjf7nBwQgBQpUqRIkSJFihQpUqRIkSJFihQpUqRIkSJFihQp"
    "UqRIkSJFihQpNhodjXf3fnfn7oyiSsOSYlECBYgD8VUDooyIVqJKVKeYX0dVCjUGo0FmLMP/0bNJwalOXKeYDWBL6MpqPNKO"
    "8tvxdDLMDOcuzUaK6Uj2IxfAbtATviv52d/q50M2ORvSa7yt/FIiZ6MkAzt+XXAYd9bdRGBYn3xVnru3jDLhuSIoo4/kkE+E"
    "GPkE4NwOI0BfIxQ4Hmbd0UxjprsB1L3JRhAd7oC8K5HC/xTugedvOVE4CceCZp8QWzm2fhCmoZlliRMJM9/Ut5KMMry9Ctlx"
    "cDt6WXmTY1VEPFFCeqL6G1BIdPKsZH9WdX9PGJ3yuRm703lb3k3kE3c6uwFPEFT4EUqfiB4zzS6/8baWJPoe9p46eE445o1K"
    "ZsFIYDcxF/hz8pb8zDv56UCEbRiKdEQuCdssRCcIAzNYZivPQFiCsUmBRe27IIldbopUR56odVlHQ5HsZTb5emDhWTzHvByb"
    "Jn2E+umK3G1O/bT93Zu5IZBkukekpH0ckBettfm26C/7YTHMM3EtXmPskor2/rtuw/vzzro7TQEsc2gkgVraIqTXUb7zlPdm"
    "UGcJ6Ikq65LJ2EphCagSYw5tItliJ4whVG1DcPEZzF7TkMwGmHJOl0CjG1qg6CQ50XgGk4O9u4+I3QKLnZjFT4Ce9RTXClbU"
    "knasPE/kNhYZbbUeMyJpDVaT8F2A44H7wTpAb+2jaC189GLVOSqqU+WDW2ZZ3diozC+G5PiRsO10d9/b5dVb2z1cq31xgzkd"
    "0t1mmVaLPtF7+m5QxOWVPg3bG8fOAodVZ7suwEkP+BwdWKHFAs10KjOD3bCUtvbFs3UQ/O/AUQMZs3OahOcZtGSqJxsKmbNy"
    "1Idy9Gt19jLKASyG3VwbRT2HjwbDJGIkz1LJe/Ol0Lll2brPVJ/UB/S+y35daN1YzxtBaLFjGYXestHr3zaxisqXrobj7JuI"
    "/qnZdxl7xGBYcAYjgB5Xu9vyJATRLI6tAd4VKhZtKy6JkK/rlGsiZmbHSzOxyYWxVwt8ntkFerstZv1DoPcO2OANDf+9A6Zl"
    "oszSkm0tDTQIY9KacUyRbo9cXtmt2dBzuMEsR1HfyhrVP4NjByButKYY8V4stQ+cU7cHj8qYBcMAFZyo2DH8HCJxO0gKXjyr"
    "+hX/sZAEnNsVsgXwxuDJG7KyMMgqxA7zAPshT2c3BOiIs26Ekyz3bsyEs7HPnspzOOoSo6RXefx/ctjuyW4Tj3k+uz7jHdxW"
    "EDZUIdB4Rl1NPTzP/Owt/FgqecOJHtLTIZ1aZg7D4dLAEcdz9fIcAmznvUfbw/N/j8eDSbV8G7Ih6MdTPB7eM2YWB80AfrqH"
    "GWwL2ge93X50hb6M7aby+XwGFToiE8zAj2m845iLZgc+I/MVO89B+kvv6Li3S55RICz+9z/Gs2ZHbLuHB31Zypi3uwcMsQwu"
    "mjnQxq4C3gCAQrxr4wISg87djheUq/cCPaRyNwPMSvLkjag8CL49XoUgDT/cKuN3HmgoAo2JTjMm+fE/izwrFrpV9BmDIxCD"
    "/+Ka1oIFrLPyrN+lsyi4EpFLT6DrrUNLVjsyKxwi0Csosxtps2eAjpv7sQJUl7/xEB1HMwvOs7jxHwP6qfLcuo2AZXvcQPln"
    "G+8AyLPaHoyeU2O2dfLMt0FsHcugvGumvmVW4OHZ7+xDge7mK9Ep4AQnYraMPqoY4aVe9+8YAWs9wwM0IlpyG6SywXCWPyT/"
    "CXlu3UAbH7EegbTPONmaB30xLT+w7SvwS7C7O3BWN9GyH4htGwjw06N4nrHW3BIQOTHBlQXBMLwcX1wL1eceiD04Q6AfRJ/J"
    "FD/xXf79+yc5Dmyleer4eFiAkWHX09Q+Umzq8dDTWcvbqyrl1anxcbmiR9uDc4sf/3zG1jAcKaRrfTqU55lXTgKW2py2g12I"
    "L7UNeYyf5n8unukcPxHLJATQTwSzLNC96iwWaCT584CtzHj4+1ZwVjfiwa+zb6LYNDarD0QzjkEVjMB7huE89zpgpNNQiBwO"
    "5bQ0ozLyv//+s1poWsrAs/lPJfpJJLolF+x789Ysf6jAPPqA5hlGL8JChKmkmnsG295CUnH9LUjq/3wQqj2LWpE0d1EV6L7S"
    "Z8cKBqNWaeMSwRmBHK/Ptmhprfmp8/wP0fwUtppVdnrFmVsgdCt9RI+VFnPdo8hx19u99sPtlyDOyHQ8x6mzZ8Es4LykQxtf"
    "R62g9Fc22q0FEtn/XFDjVUAs0TLQ//i4/UeSwv86nYzeFijKsVRn0CoD8ozFviebkxiMM0fa1YgIQpdXPE0qdp6D2j/AdDx3"
    "NmodAg0e7TZ5Rji30sKgCfQ/S9A6B0X7P7nc60pMtQUYtWz28Cz6BxGt+gi8C2CVHU/lMohmtexoUB2zBu7sqLDrc9h5Ocfz"
    "/HKLsFOf243Gf2qpQkL6fzjl+2cPdj2tRusoPBQQXEBKdsPbzkHGL3AMjXVwY8G0j2hblvF01UIA8PZLR68ZxS+ZtGPVua9u"
    "sUgiNzvkJCX8zwU08dLYc2BL0j4GvA4FyHMsjkp1QEc39vvEcjykQoZ/banX9+tAt64jCNXP4wCg4+SZ1CX+/PmDHvpnAMtN"
    "07S70+BBRCOe6SqKFiTXJ7UO4qI7Pt201vCQDFnXX11zzm9A2tGx/ihcNVHQUyXXt1baQ7P1VV37E6WpMNgKGDxHSi1kekBZ"
    "w0XzsWA2Ch//FKhp7eofLUbTCq0EhuhP6uvxC9xHemGWKuAGftYeDyfRcQ23zj4cfnAnADGnDVR5XuDAk8YlOtssXcxTykM4"
    "s8oV0WedZ0dJo1UoF2la2zvRvTCLVXRHp5T2+RkLtH2ND1i664yjtQJSu2nXT5xj++eP2QcvV+Pa9vhE/6M8y6sLT4VngYcy"
    "UMC+rowyxd52/s6n7q2EsahkC7mht4a02gTz7HUNbkWOOPYq2DyHGBG3ONvHAxyWZxvRlGe0vvBUeH6GdZEoidm4Yr+9Ve8h"
    "vwIrc7Oj3X1m3sfz076cEHtQWPDRse/Z3UZDe4+MAXm5R/31OhzR/wTQTKefJtFINnlTgJ5cdG0bWVdy80ztBNPf1tkoPeWQ"
    "BECpfcZA5PllZuoJlVNGmAna5fn1chWlX0fgWYdaxVlG+dk+xW5e2QHrgt07W/pe2t61QQVYa7Wzt/+33VI40+5WmWjpqyum"
    "P9h9GM5UoC3yjAK8SEhv+dUq1xwFaVbhNWh2z7my7AyiT7XhPYIBYJzxAqyt81E6WEUcCzAnz525oKkdEjAyi5vOSjc8oSaY"
    "vjjNLyrR9AL9r90Xzk7/a/eSEsoK0Q+DZjJmQGJu6FQ97FVSp0cRx6zQ/f80QAew+LDmj6ueTqppHKXnlyX4QrgZLUOfPnqr"
    "NHsyOu/D+GcBAceQ6SQEqX6246dEO5ykFy5+MjOp7enxaCc4+06vfbbU99Y8P1qDwl5+xvTacX45w/ExUKDfAM/PsbPaEUeh"
    "EEhxbqeamxZw79L3rHLiqKZocwPtXT9aGmXQ/LHgzHjVyQWgF+egIPWQ15o4RyEdoqcM5zmO3AQhW8FBojw/WO1hVs/Rde1m"
    "gEbK/OdPY1VoBi5htyUWGf9CeR4NtazViwl3f3o3AGdWJesC4fcWOUzXAUDY80o1lYd8HPgU5Y54oFewHIznPw6eG0hZo3AL"
    "JM2ejufF/Iir/CbfMkCdWRPQYJztQNOPCaAdpEFijscPEs09x6OdrHxnWyD0HSO2pkDbb2VFDOE3XlPYDY1d2/VzcP0MDPUh"
    "QV7jITU+TzU5uFQR6U7QlFHHTY4MnPRsrNZCW9+C+fLJoMttNIY+T2SfTXiXyB2fUcEeE4aznBdNZZvFwmNE1RXPGV/SNop2"
    "rymZfgTg3E6fUoTwbCUajU2DB0hOBSfnOc5pDy7GPZ8DgA7T0Yc4e95EMA+2Lk3DiJYWv6ddAh/QyLMwy06e4dig4QE60dOi"
    "Gi7jYln9FVS5kBaouwE8h5XeeFdb10nzPLzpEXcSDV8XQzUrHeh2Fp7j+qiXxdlGtLiF3viaqKwR6DZ6CtzyKochwPo10Tg/"
    "g2fhQdehyT+RKw1yjtSTjptWvB724gcc2OI1hT5HfVqW8hn4rxfoiYoZ/ThX9AcOca3ym/QohuxAQxFmOF5srbmX5/FTNkU1"
    "4M8fyjPriLZ3s7JhtFPe7+8si5hrEq2KspVnfpd5fbFCrRHiepep/vn5aRnZXTc90KyrsO33G6NhnqS4BYHGk/cynbNU7ZCE"
    "hJX/1bXdADpbMK4+Mk9hw+I4VB+yRLqnQuxA25oiQp4lqZ5BoflU99TrxkxX13XTfdE9aZUkBzMznl+iebVt3VXSNiYDB2Dr"
    "PGt6PTnRv/Dv728fzliUTaZfpulQmuWkkwlOhDNUfsHzyy/Q3slyfv1sNynIIiXkQDN8X9Iah49n+erJcW4X8RsmzxrM0+D8"
    "K4WhtlVAGDzzhFBO/rQD/SLgtfH8lHh++Xs7w5b/luymbKQu9kpZtWu9Jf9QjQaDq3xz8Wyv2s1iNTjIIoRSV6EhxpzuQc1i"
    "xmCecWjXKIbSv/oRuqo9gx77gG478+zRlGdZKgaV/V/b0WdX4Q7M5JurvnQvgmdFahjPL47zsOoc5VkHmha9AStzODCNWZZe"
    "EGd6QiDao1zxoes0nqsqdiUrcpUr5jM/F84TOQ2sy5HYenmWDhxhSgz3pj+C526oW7YATV7OPx3dkgu6kTPbEqS/UMZaKcda"
    "vHxOzoc0tSSRurwUzmqhThZnsjQ4DmWKc1VNyDMDmp8Tg95ml2UNUI9fdgl0iw9rd8/Hhr+kHsN8bVg3RyX5BM6zWe93LmGF"
    "Ww2y+tbqJ08GniRk8rdu5oHjeY6F+RxsoDHKVdedKwq0YxnwRwa34ogr5JJfflSg6W9CXxwVu+3STIm+Xq9oOolA82PiLB6w"
    "/wAMbTGmx1njQXvxBon25c+q5138Hu+hfx04n3GY19Jb+nhmK4MEaPw49J8CcWVXZwT+D34aC7kKzz/iFyLPe+WZE40UGiP7"
    "BQPjGVLid0BtNSMvcVvL9mpSu0/bn1ZPE+Tja6e5ie4j+mUFZivOZykcV589Ms1EhSWDVNPpowyan5Um1tJTCZ5/PIFzzdeL"
    "HVrd7g9nBPQVA/34Qu/7C/OMkKtigXbUrSVfLTWMKVLcSqy0q0lzrzi37iLzy+Kcz0a4rneptLaThH5DPFwCWsLXxzOj+fv7"
    "W6X4m1/zlPQZ7DSQ10A8f5E3ToFu7UBXeLXVlxO+3Osw/IfuLNRl9LWAbnx5ngk0wfdXDS/O3hA2xMYzuw+xHIJoXnoWD7LR"
    "jCeO8vyN4+dbgpkB/WQnJ9gzz0Sgm68vMpiE54rwbCzE0kHwMB3Q+Ph6CXlWqW4XLXPYYBZ9WQbNKtCCXTvP5xEhnqXluiIR"
    "Le72RIpNksK+zBLTKvFLJflb5ZknNO0uvQY1lDCg3SBDhHlu0RIKZBkVpS04Y6B/x9dqqXkWPIPVeEab0/EzVWpNGD0rJlag"
    "p+HZfDZLikmrHJE4Y4RltNFy95OfsM80ft1eBBrxDCEmOFc4JcRL4Ahoy9Cgd//7a2+0iUd6uZpGD9Aw4cJIBx0/AocBsPFA"
    "T6fjfD5PybPldvKD3bdfn2mF7lvlWf5fkWf4Uzu1144Mh8QzzgUxzq1zfRa9fcLzREy/wPpAY5zv9zvcmvud/MQXdccBfxMU"
    "z85zi/9a70L/qXlmdZbzRcue9VsLeM2PsBs/zqlo98cz9c4P+PdBaG5bS1aIZo0B/TuFTK+XBvNlFLSu1BGOcQCAQOZY//7e"
    "DaMBBM+q5ZjQbnTYK1fee4aqvAtnpbwhn91MyWt2lQ9i//xFixvoH1paYTi3cMQ0qLWU/neIkwYr06wcjoL+69r7/XJ5CZ45"
    "wkyuf3lNg/MMiYZ8A3x3GDLQo2l+tZ1czBhnW5g6072r6pp/+CUgrwjsUpp1ngnUD+Y2sD6fq1ZpDoP6bCAdq8jrsqz5jcvl"
    "go7WuV7hhVeDeQYAXuREK6VmiWapPMM/AOOANlfDz914oBm2lcIz9BdqIRooPL+wh95h2e7K/UbFStAMaNVwvOz6rFvp36Ci"
    "xvI4k4V9led70yCKL5crohljfIHX3skle3++zLMUd7ZgFE+03fNSP1xJK91BPsMEGi9uS9mPKs+s/mrgTJDeJc+PL1kWvnjz"
    "ANFnxW18/zpCTLx0SdfudT+1RlXj3mCaqwuGmVBMLQe5bCHajjMmGndZwA/0byTQf7UcBZVaUNXihy4LdgO9hyTPGswmzzLN"
    "YtbAPvX5i+mz3E7wepH8WsX52wn0rzTpev32lxCx8l5IRbrFZkOiE0KN8b43HGhj0ZCy7iD6jsvYL9rOFUzd30ohGsILx76j"
    "0vysaGvdQLtBShcOnOEP8PPjOnnOLtcHqd34spTmzPLzb2hovoOyvratuqL3gDAm/2PzLN8F/oK0GvLMMG8uLyUz9PLMbTRt"
    "6AqE7i8M/EUgeDGH1TZwwHt0/hpHYDLoFGfjPfzSfH2v690iHTSPyMQFzWFA/2otEPTKDfBcYZ5R3FFRQ9FnKsuXhqd92HPQ"
    "Vix8f/6IOxZv5RNx5+h3COgw8hDOX48H0hScsHS02XmS9Ril6+hbW9620KxX63adD4qxQBdeNqGO5ll2Ic1r1beJnDKMa8Pf"
    "jcwiYpebaHJPUehAH++LFPc77/lortCfQJQZzaTVFgl1qEBfLpBnkom/mDhPUKjDptlWdhYLKFae192PTjTRCGilQ4DkgxWr"
    "c8TzrPjp39UH6YrfJ30XEOnGqFNI0osuXyG0F3zp63Xn/F+IyYY4f7GkA/5sSJetsuqCgA5DD24W655px5edpTyQ/G/HmRwu"
    "aMf5EDx/SWNBLryoTg/neSMs47d54bILCcI8AzfP8LfrBes3GhvysAtVbvwsaoHzi/WNUylntfoQ8C4VP1Tlq63GwyyVnfWi"
    "BgCKPn8DhzwfA2h5MM6S4XihIwtfQ3imP+8bGB/8Hi9KKkCWrC+ca5VngH+jAo3uVKHfaLFayze+5EAZBwVaLS/bvYa0ut09"
    "JqQZOY1KMRsSzagB+tsxUuumOZNNNQVa5pm2l6sHfMfKc29NYMQa31B5Zjyjv5VDnynQ3E6z5RbC85fOM0zp/qIfDGhmUHwy"
    "Te9Avlb9EVtlrngFxAb0d6VngQDzjN8Rv+LA+iwEmpPMRqdV+qCj3POMNPOFviCurxcL0LqBdtgPngXySwrPGOG/JL7+Sk6E"
    "v6BPSlFZA/WBddE8V7ymV5n6/E3k+ZtaDkA5Bpznn59fN893sH+BRouE6jGazEWrbXZeiO/8B8I5oGK7lEpfLxedaOd9LxdF"
    "rAXOwj9/SUTj4v3fv+j6v3+/vjSeBdCWI2FZ/GU8n8NXYaSvkHxWxpMSnr/FIrf0/qhgvyxA38F99zhjoCsMtNlRq1Y3vDzf"
    "pUD6LDdgzh2Xa+89LirRp957Wh4qgOYS/XVBf1j8/WvwjF7POBpWr/V/Eb/ho9n8OMhEW3iGFvr7W6SEQEBNgbbYCtIlCMBB"
    "gK5sh/W8XkbDKIWW9UiqLJMOS+m6TbzFiwL0Wb8VXHTHbHgPBWjkOVCZ46IHMh0m0GXJG+YUnE+n80kR6MrZskTOVKBDjVgm"
    "J/2SD6ZFfgJnhN9G34ZM9K+xunn/PQTPAK+cXRjPcJQruZsLJoMS0veQkNR5Q3svxtfZArkiyy6eNaC/bE8Aka4qh4emPTKl"
    "rA8nhPMXM8Odf8FPSwDFEd3ykiA9ghv9MEt21CRbeUarX5uasDECDQcfOugT5lk0LNLRfMXxLOO8LZ5NxcZVBnbh4vHU/NJF"
    "FDvwD6S/NJhGy8BeKv1IKQK6ZDfo6bwwsG+FaarrXcf9hWw8yGM4ztAyV/QIqm9CsQ1mDLTM850v1pO9KjgA0GQ2TzAYz5Uo"
    "dyiHed9jA2wJaE2p8T5f0srSVuqQH6oJN34myPHpJCPNgZb3c6ez3A0jp6YMZ3wGpO79fiPb8X6zVme5RC2C/8pPe6Qsblek"
    "yPxtx5kI9F1kgTTnIV1X4AhAX+FkVmTYid/4quxn7tk1zpiiUyklhucTebuYbEil36uoV5RMn08kTKDpLWf6P28MRT0bX6j0"
    "/ICcwksdNC/dizhoVoZ7v+VMUudZ5IL4tDJS39EPV2cXzeCOcL7duW3e6oSNqkJf0Y6RzkwlAa1RfQCe+brcmQJNd0sIyQBx"
    "FzwDyjMDmkCNeUbk/pWvp7cSnFlhD/H8RfUa9TDBvBCONy91KM7ZF/zUdFSbEc83heebapVvMO7HxZkvq1wF0Jcv63ld9woz"
    "k9jqpNa+FNyGPGmlPQeRfwg0Ylq9BTntE12EwdpMj0LG0n5D7aikxNEJr0yOUenjuZN4FtqMoGVY324qz8Q3H5FkyjP8hyZC"
    "kRLaRzqY5w3SrFW8dPks45/VeB4lLVSe/sz8Ol4TZOfARP8jnm836F9JVshKzATlXn2m8nz7lpudb0rcVaAtVgMcLq6VzPNJ"
    "aeYYAPT2aGaLGy6c4yWayLP2RDLQ6osQoM98nZwuv5TYet8w0XI7Pz8Aq5fnrkM8k9BIhizfb+p83FWrAQ7IMolKFWi2a5Zx"
    "LveA81VbMLzQWlrlRVkkdfpHwYW5nA3qT0KA1l6J+3W60P1XZKIE6NuN1S8I0yH+GaeEHGeGsLhon5DDwwxwA6ViOHjVNFqe"
    "t/W+aOGByaUTZlGlEA+lpiGKZ/oshOez9IIy3uTYQWG/KdEqq7JIe3iGfkOT5LtEsmtSjmkzVKDF6KodBzQ5L8v94vxVeUG2"
    "2g5cXDauZZecOBM8L9RxnHn9RPkowSH++1cCnEm0zmtAkUPCGQo1/Bs0K4fHGQEN58rk+Uz2iShv2SPP5LCmCJwFvZZrJSJP"
    "nufEPCOgT6bFZntAzHOlAV3beLZjzC783EzTnHhmPJtA0xx8pzyjbVdS3XCi++7h+4wQxwFHlAj02Qa0Wu1jCl2b8uxNCHWz"
    "cQvTZ3B8ngHcIwIGdFnSPg7SBoZ32/u0z5XbF4yLc48T/yu+z6qy8YyeoFQeQYCuNYUOxzliWj4AZyBkCZvOK6lAn8hO0cmz"
    "GKAt8nyai+YecccjKL4ZxWS/sn0EKJd1V5vkMq5/lGvrPmE+bo3ZF2dhGDnQ10qWNlWhpbqPXgfaymeT/jgvrs3kpRnPFGjj"
    "UWebpnOllUWaN9ERnGsF6LruVeYPxFmWaMbzrabzUEm1rNI/OusP3ZWsd4KTNaFbTJ7Ra5MBw00jFRlEieGzlWcBtOmjmR7X"
    "FOgfynMdgvNn8kxYBnhskUbQNdqT1BN57zkv4epDV16vZQlC8rnZgcajFvwYMr4S0IaVZmrMr66fNeE5Vp8/hW+R2sORvVxr"
    "tB4rjzbi2ff49bUAoVSeFtBm0A90BMxigNXSm0R0zc0FQb3+qWn0A61P0yf5jdOJ7i1v2HGU2oBvnGftozmnAg8oYAfwrAF9"
    "QwZDBpeDrkRcceOzDDSZLO44dIUGPUAXOQA5jEPj3G/PlR3eYH12BuM4iOfPNBtK1Q4CfSNA6wLtf3RRFITk4tA4nwIsjZaW"
    "DFJnFlAfCg/P/hrHx7oNUePAjoOnhLvjeaGcLwzoAH3mNLt59upzFM+faDlIAbXAQPOjhW4BPCOgMc2fwDM4hVmOvm3q9RpQ"
    "JCDTN5d/9q0O3lA3/0czLVYECgw0TtNRIfUehDP/URyb57CUELgaThWcb36e0U4Phw40/+EA+oZpBjfGc/bROSEoi7pmqyrn"
    "E9HnsKcgWBdHxjkgJfQe+SLhXBSQ5QKFleWCE82u4mW7G7IcZvuG1NKBcL6h7xEA9+yTJRooQKNGjrJBEhL4DAU4uDoHOQ7v"
    "0VyS2SgQ0yx0mllQnK3Uu3DG00V5/nAPTXWWA03SFT+nMAsvimJlqKfrwxjUJB3MM30sdhoyt3aeqTzr90AZocYzPgJW0Ix4"
    "/nScaf8Dzu5ugmcFVgvMBGf5XkVxXHmOFeiTbVNLA2fCK4G2UHnOmf1wxl35jU1CvW2eswWNkAK0n2d0oxh/rtDlYXEe7TiA"
    "hDOD+kb/aZFLPBchCy6K0mwa5wLyXCyDdEl4rkvBc+Go0AFMM/1duV9ZJp4dPJcyz2VhiRvHGVCeURk1BuZdhF8ppxZoGCVa"
    "Kbzl+suySdAMs0C63Kk8n6ep2fWd3oPzbLMdmi5goDH3Top3yfKy1rSg6QYqQJd1rr2qOejy9Xm2vN0AC9IcxbNjD4g+8Ixn"
    "K9EGz9xfa8aDpH47ztmW8NGowMEL+FVd39UXdfFMb0F3zsA+FXpqw+HfB4boMwKa8izsCC+E0FJGsV+em2YRoIta7llsmiIY"
    "Z7zyXWSI6+y4PAe9on+MczfPegmp0PJFivPSNnS3POMxKosaV6IRz4XJs3chhej0gkAvzzMYyXPu5Nle4DdxZl0Ghb1P93NX"
    "Bh1AF2isIdNFhSx0rg2tzxLhOzTwZ5YfkOfTlDz342zyDC01qnrkXnledPe4F6JJlCUa0jzPbKsn0rhTRaY803uha+fvVDpt"
    "D+ieLXbxbL8rHH6tMn275ZlUObXMRp4niq1EQ8eR5YhnFHnh3MPBmzKzEiNgLo7Cc8iqd+8mW3l23bXINZVGRGdee54XiWcr"
    "0dhC5xRnpNHFoCOrsul4Pm2CZzCGZkWgOdbAx3NuWT30PHueJ8PhQLpmA465zlc/UpAcj74iz/6aXdibYAlhKXTafU9Cr2U9"
    "3KPPyUB7iObegwj1sNr5pJ+C9XguUZuz61WDt5/zzMscPq3NLXmhB2gsz4lnT+2fkl03WcaNRyzY2YyCvSTPpUehwTCg+7oY"
    "gZXnm5vnItmNfq5zXI1m+z4EdJTzyKYkWP5NcEG61+alGb8A5fk8GGeN56D7G/LsAnpINph9olBDoslYkl6NOCc93R7QhbME"
    "Nj0v7aRwS89uO7tY5JuQgC4G8uwBOvmNMImWxzNSn2eTgLI3VPs7imbceMGArobCHM+zDWhX6RRNVJaWVGKq0gToOJznGt+i"
    "DCFafO92gAS7bqIrTOzg13IwzUqJYxjPniJfUdD6anIbwUxHF+6yudZUzEmOUmyLobDczmmujVsHbnYIz9IIx/BMqqqJ5yiC"
    "1j9lnZPnghV2g12I6zZ2iaszatMq3TBH4I17QUmfjJtnVZ+BvvTtXFPM4UOz5Dkigc6cBTSAgVpGAgp3WCgOku/Sxjmvwhel"
    "7yQbcQLt7+/KVfzZz0Jk5S6a80jHkYB28owLWqdy4U+XB2r2owz3I/YPAl3/t/I8pMKR97UrWnjmDyM/XS6F4px4jqjeWUaL"
    "8gIUW0iFOptxU/IiPqKUW6K5rkt7J8cwBx3Ykc95BjLPbtONyk8J6JjanYVnujheUl+I+xkh3OzQwtnC1uAQS7VXuumdr/UV"
    "Ic39CBiVGPbynKv6bOi6q7pB7pTlCedIfc5s6pzLe3m+S505eRwOdKH2bVpEW775ikSaFADLkSsqomYR+AbDHsZuoA46kRq+"
    "k2fDxYaaIQxsp0eZeWRHSLTHiOg3YZxrfRly+JpK3nfOHhvPuIfUjTO/f3LQ0dOBxgt3J6HRFoXprGgMoLMlgJ6MaUdcrxjn"
    "AiWF4+vQdGtB4IJ3rvDsx5l8ADKQeI6ZD9ZsB0SXErYhTVdYeM4W4Xl+oNnpjcrRyyokpQtSaKXc7zRv8hOlCkd00IMKacst"
    "pQnibAFhdjc3P8/Xa454pinvJMuE9AwP7m5+9e31D4KBc+J5iMSQ7IMw3RRWpLK9A900VxRFXU8lzyIV6RfoHMTxzArQxpAn"
    "wPvGuZDHL8+abB2e5wY6b/IrTAiveEXFKHCMs225i+fM4Tj67o6rqYnnEdYD2WY8fl2nEJXTg2iX2P/Nqs/0ma9XmByMr9Wp"
    "Q1cE8BxgOcTo0pNHZKPxzT4e6GtHU0MlsmxDQA+DnuJsM8+nkUMXWrLzDx8fXX4ylOSfx/IMTWZuj0yL+UouomUHN70XShGL"
    "7C0Gi/iVmOdTOR3OPSIoi6533MS4sjslnEfzbMOZVvU0nvP5tiOXTnGvn7BzoMdmZxcgf6e0GyFviHDpF4JsmV3gJ/HcQXl2"
    "CnNeZItIdCb7Ud1wBgBtv8l7PABYhmefQCeYpx92KM9do2AsiXLOjqHNF5DoHovpBNrNutzYvyjOpkDbbIpbJBLfg5OaLocK"
    "7XDLmZwj8rN4rFVkZNBK9Mp7FJtISy3+2sErCwxtQ0B18uyhOfE8VETyJm+cOGd6zSPLVjtgiy7QC4udF7krbAKtXlrKzHGg"
    "TdlN6jxLNtg0tDcpy/p5XvdUeFmWe6Kw2WztjAfMSS+xrRxoUcEQtAakJInqAeYZ0kx5zqw8m2niikA7eRZbys+oqgh0qR1Z"
    "uMjGepntzbG3mSNmG+c5bxpxZrsAntl1Kzpo175ErcpIjkM+8Hs5nE1sXSzrPNMsMunzMHn2h6x9a/MsHSWd+7eaH2JzvRba"
    "kd9FueAAAxu01i0G8vogkFYNU4QFbhHt+nC26d+a8iyX6HzbypPFa1FUhZQMFkt+02JmFWgHz+r9Es9ucu3XQiCaCJyZSmer"
    "pIT0LBbawWBunHlrYFOgblHO89JflBaoz4AXQbRmjhQWIS4cPEfJs2GnV3knQUA3tNUVHabQXK9NwwS6XPp7nTNrVc7qngnM"
    "qomm1ZHEsBLlqTA1+gR56LpsCM+C62JNnAun5WCH1mCem2tDT8dYLo6zCrQnH+Q8c+3Rmz+SXMtEW65CPA+TZ5YmLo200a3h"
    "ABomuby7FB2e0tT8i4pXyLlDeBbX5Zmiz/TqpkemE+r4dDEx8pzz78nK5arHoohohxsUeuWO/tLInc/FtSzrIuIUt5PzbKSE"
    "mc1WZ8raSwYy5j+0dDGRa49TGcWzcwVj0e+i1lr79b4SulnKgb1X+bQcq1RF+0sc4hpdb+X+fk+tOgk0EuiuGZoKKovMAd+n"
    "PKODlj9c5HTtyj3YaWSKcrWvgM+A013IODc6qxrp3uXEpNwwIYyr1mW9vT8rAK202DHhrsV31eHzItXFeups1jiko1AMeZb0"
    "O3fA70P6w/3GmGKdo994FaDJCXHE5kg8l1FfRzV/GRq4hZVKsICTF/BEmqsu51qs+ufiPIDnTCRhNpzLFYF2Hm9FTvRVFEWx"
    "Ks/ObC7TM8fcr/Buoj/bcpxOpyH6TMoGVp79XwC8EtCE5xKdmG/lr9norU5gnn3LVBxpZ4/TR+vz6dTEkCyVMwrrodbSF1tv"
    "B2he2cjX5TnrPQ0BPpWrY901s/RMs/RRK45kn8xzb0aYh2SB0pf5FOUmecar36sPOWFOGhyVXPKByy3dNrYksjE7Iz/aRGOe"
    "G1u7qLakHUZzQc9Zv0l9run5rcEWkKYCbFFh3x7ELIrgs6agf5niNT7XdRCgi8CiXATPs1vVMvb8G+ungzKNtEeRblQR+Ehg"
    "AJ3xA25TWkiArmttSWUgy1SWF+O53CnPEo4CaEmgdcHGQl5YH++pQ3+sRiOgGzPli4ZZSrvKJXjuF+hafOlgWdfFhoCmSLLj"
    "093fplJQopX80DwyrrHWtD8T6dOpoh2jeXAEfIfaAjyHCHTNFgi3hTOrZhDXwZbpyXcvFWpyQMhWHya1FRJ0G8dKYbbW8YfZ"
    "mkB3dTMBz8WyOMfwXN+KLfLM5j4HDGd+MhyW8foSYv6te+ILQTId6A+sdUCeAwS6kE/cEoLzujwzXb4Vtxp/k6YN6E3RTRmW"
    "yze9BR7+5WS5aAoxT1az0qd0PaAhz7hkZ6qwRnAMz2B+nks3zBjg2+2G/pPsxjYFWrISbAsjtzTPHAdlfWKhAwu0zLM4a7gF"
    "4Lzo4blYqGfejTPh+YaiuGGgl2yWmkCmA3E2vxAr81X5PgfoM3Ycescl93N5RFVsqQYOWuCQlFcC+iYFNByFwvTmpyPsSw2N"
    "NBFY/IYC9MeATQV63Hfy0K4NzPMyXRIl+ZJ59C3zNSH2Rq8RNNd2fd7XzCqfw8LNMzOuH3+cChboRj7x2yCoS9pgt1DTT0ng"
    "rW+CYHqNBHSB9Xn3PLNqtIJw4QI66JejC3RXNGOBJkwvtV+TeCbaLORaxtnGs7RAsQMDAuwJwC7M04oCjb7ouMnGAV2WiyXU"
    "p4KpMQK4ILajoHkgdxuiyKHzUIAhlYTVcU48BwMNgWCl6GywQC+Gc6maZdVmkGtuhYdnrUK2O3kGM2a3u6/yYceBcaZED/oe"
    "KfhnsZEoud0QlQw9qFoTM2LlGXdQZNsF2ZTnYgzPH5MpnojjoNHwnoLo70VbTp7Lor75g3sPZSM1Yup6q1rEekm9PEcS/aFA"
    "1wUrR0fyvNghIKdTWffRzMTZvaCCi9Ub5Vk6vlgcnGmzIYnmXqDrYnAtejM8EwNScJ5v5vST67fJs3J8sfTLKJ4/afUbAn0+"
    "VxLQytcPbywf5N8b4aaZ92Jjlm+OfKoo7pucY+2ATNFcrrYsxfL8URINeeYiXde5/h0kW9Jn6WtQXDhzqEtxWkYLDZDnLdOs"
    "B2smFe8gZvXK8w21g0sh29ZoBjSqHAzhOVuM55M+1Zo631Sg9eoziy3aDXHYmh1o0R8dC7QdvwPbEOQ6uq6qqq7uaonnfHs8"
    "n2zTTUEuSlKSRkC7i7fYXG8RZ29oZY/bbeTLjZ6yTX8aINAVDizRTbEffTblmpSnrT6Tw7A/nmleMGwHY943P3iSiAQaA13X"
    "VRV8FDXDZrHyRh/QbM2QA21xqGgJ8b5HniWlhm8zH6muBy96nKlAI6Br6SwEPTzfFm0vPvUSrfEM7DxvsLrBvo3ZeEcnG9Bo"
    "4EcmcNkeU72oMgdFGnf4lIFAL9wu3w90jU/DaMcZsPNCb0+e6XvCb469xZMURr5Q3qJGPTsqt16eCdJQ3ZS02iPMm+MZncMc"
    "8kyO7QZ74Zm9K4Xg08kNtFaV3nmBbU6eIdA1q3h5BLomBzXRA0EWnvleD13acfbwvF570skRpXFFPNBSYpN9NM9lj0CrHRMr"
    "8kwv69Nttxslt88qyyudT/fkZNlBuOWj28dzFoZydmSeq36e1R62NXbNcsJkmW0vz1o22DQbk+ZgoG99QGfaV89+uN9gQJsN"
    "HfRIkNvih0uYM+veGxslBLLlzSYm+DQk9GpH77JK/qlfm+zl+S54vt/RP8wywrlZ/EgPMrFVqVYAytLwHXae4Xu7bwnnshwl"
    "0rcYD/2RPJ+rm+BZ+Q5LyjMJdAjqrVjjwCWMc1W6JvnmFGjW7XHfrTxbgL6nwwn9OFt4zvM7IvouRQP/Zquc+P6M5vTsNJXy"
    "rsXwz/DK20Za+afiubzfE729+izjjM5ncNcCf13FeiScnTxXdR/Pe7bPdoVOQHv1ueYah7wyOUOrqsz37L7qqX4VnuXpvUl+"
    "o7DyXG6k9bmckucEtI9nJs8EX+w2ZJeBec7Ahng+qTzXN9f3TND530SCVE7A8/3O5gkkE+3xG6XE873IZZuB/627wz7hrXTp"
    "VV1Tnq0Qrf7tx5P4DQloiDSZmkSwJx+UedZtM2IabIpnYpxRmZGdm8PNbLmN4S7L6QxH4tnPc+nkuaE8b2BT9dnFbYH8PGDl"
    "5ne/43g+GTgnnj3lDeJFHTyvP3Ymz7jNtRI4b5/nkQptCvT+EbzPwzNLB283u+HYDM9nBWfUdcKA/jCey6TPTpxPJ6rPVpzv"
    "W/DP5BQL0txCljHSnGeYxIIPAjr5DQ/PVVW6skHK8/pjp9qNap88Xy5jgE4JYSDPpQ9nWn5ef3NPKtASztB17GRyJwM68Wzh"
    "uZN5tpsNyvMmtlfq4DgTiZblGXwSz0RkEsM2ntn54e6e2NAOhRHNjuPdE86Q5zFAy+qceDbp6Dp0HrszO+xh4zij0ytQovlx"
    "6RX09jvCmQJdjl8j3MCawOZ4rjp8XsYT7VHbAc9yCJ73NLOcZ62eTvY5xnXWnhVWRE0My1FVXcft8x5wxjWOyorzrni+4C03"
    "VjzPwk6J6ytXB0fi2eSZ6DMZp8bD83a2+VRqPGOid2UkOdCVmg5o4eG5uif7bAX6iv3zqUefN/YZPClA789GXlSFFmHLe1UR"
    "R4uiqEqZujescb0Sw3HC68a37aszxPlyPcvLQNX+drsXATSn2rxThVPfE75ZWRTFOOMFgcSzjjPlGSndfQ84Ixg4zxUWaGgm"
    "dzbsl6uCMyTWDj26heKuVCjhXDXbWODanlBcOzxgYhe2cZzB9XIVthmv0u9QRWAwltEFxxu9KDzLKcO9SebZPmQd57naBc6A"
    "o9A0653haCKe++5X4XSh0guUVXTtufkMnLHf4DxrQIOtphy9IOwE6P67wX9EwEUth8tz1Ge5aT6G58qpz2n/NefIh5Nf6XGP"
    "5hk0H0A0GqvHA1k0mzwnnrei5AbO12icP4lnh91IQG/JYEk0XwjOkXw2hyeauLjHVeQZeo9oImkrMyVwvlKaoyfn6EBTmq9a"
    "4swPF2yaxPOmlKdRAwwA+kN5HjpiKeacqwl43m2JM3CQmivjWQK6uSectwj0CJQFz82ReW4aap+lXrV7onmr84UnZfjUHJ1n"
    "9N6EPouCfSJn29M2jubjChXhudKP90jEbHiPymZuMNDg0Dgjfa4Sz7vheZxGfx7P58Tz5idtiD6jusbRC3YI507hGYBzwnkH"
    "HrEZwPPxS7CY5wdaejKP9kmx3Un7M6R3g5SejzwyaFi6TqpvJKR34xIH8fwBA5N43inPCWfHu1R5TrjsZN6mEfQjQY5dGPTP"
    "j4Tz7oCOo7lzAn0o0Sb++VElng9vtj+geYGkySrPaf4TznvW5w7G45p4TjgfAWfM84Prc5r+Y+6CP4TnP10H/3bXz8S56z5J"
    "nz9hQok8d13Fzg9+OKI7kOKT+tj/UJyTfz4yzx9D85/Ec4ojOchO5zkNSopdmw2J5yrh/IkZ49F4Fl/ckIbE3H+lUdiZ3Uhr"
    "g4nnIwGdOvlTHDEfTMORMoBjGA47z92fP2j1MDGRYm88k1M/W7U78Zxif4bDdkqkRHOKI/Gc4kDz/GFAE6bTvKe0KgGdIsUm"
    "gR7ll5PZTrFNpAdzmYBOVnlrRKehSHEgpNMwpEiRIkWKFClSpEiRIkWKabLMNAQpUqRIkSJFihQpUqRIcbBEL2V6KfYLb6I3"
    "RZLhFCm2JcsJ5xSHQDnBnOJQPNOf7657v99pQFLsHuk3IjnBnOIIvhmzjH+ko2RT7B9njDT6LclziiPwDFJz3OhAXi35tfVZ"
    "hk4DdO9P4G3S/EB/qrf9+VNSshzNgNhm/PcDUH6/50h50RMSaba8hvcl+Q1vPaK3IdHMskAkzkdnGbwnAdryTBEh6bp2te9B"
    "oRv74UR3tJzx/jiWBSjv2KdaK5JIB/BMce6OzPIk5Lw3EclQhOZHBxyr6ah5by5sqeinY9ypc7XznZUK4qQq+N5wJJLlqtKB"
    "BkdPtyZi5L2P+GSS0f/dzoDu3zQ9zRvNxntf8dE8O2Zr4U1RSX3TIhb/+wZqPRdIvpFWuGbhjmzMe3fxiYZazibWIlq8vPwB"
    "iydzLgndoTh/rFz37YyXqjkASyF4U6WDA8TusvtBIPpB2mwNbUGa38fAeXcqPXDT1+R586iA47C8M5y1RBxo/TDW3IuXad27"
    "+Rk9805yqsTzDOsB/dWr+NKTshq48BC8U3yYPBsCoZSmlkJkpqpxYmsFkt8rivKbtwwCBWsg27oFkJ7l3SW+ji7LU23oLoYi"
    "EXZMnjeeE4FknRPKx5nQRHDi+RjzmkBOPB9icpOtSDwfYUaTQT56pW6SaT7uB/kN3A9LCG1uZqeSrR28V+Odhrxvzych8bPB"
    "qKrhMGtzvH2gzdbNMX1PCZ5jWUrLM2zy7RmbGf7Wz56hOp8TOtuZ42mzIb31B2z6g8t+OaPw9USFffbf6GkSU4ct1m13V9zb"
    "4N/zDlKN4zNx3vYkV2QEzsZ2nj2bb+m/ThgdDGjDoH7ASCR0jizQouq339k+KzbaeRBrgvkTPccxBsRGfYL5SFC7bOT7KDSD"
    "d/ITH8W099FpEFPsVavTQKXYJ9lvpYChY11VVRqpFClSpEiRIkWKFClSpEiRIkWKFClSpEiRIkWKFClSpEiRIkWKWeP/a6c/"
    "rw=="
)


@functools.lru_cache(maxsize=1)
def _grid_bytes():
    return zlib.decompress(base64.b64decode(_GRID_ZLIB_B64))


def latitude_band_region(lat):
    """Return the legacy latitude-band forest region."""
    a = abs(lat)
    if a < 15:
        return "tropical"
    if a < 25:
        return "subtropical"
    if a < 35:
        return "northsouth"
    if a < 55:
        return "northmiddle"
    return "northnorth"


def koppen_class_id(lat, lon):
    """Return the Beck et al. Koppen class id at WGS84 lon/lat, or 0 for ocean."""
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    lat = max(-89.999999, min(89.999999, float(lat)))
    col = int((lon - GRID_LON_W) / GRID_RES_DEG)
    row = int((GRID_LAT_N - lat) / GRID_RES_DEG)
    col = max(0, min(GRID_WIDTH - 1, col))
    row = max(0, min(GRID_HEIGHT - 1, row))
    return _grid_bytes()[row * GRID_WIDTH + col]


def koppen_code(lat, lon):
    """Return the symbolic Koppen class code, such as Cfb, or None for ocean."""
    class_id = koppen_class_id(lat, lon)
    entry = KOPPEN_CLASSES.get(class_id)
    return entry[0] if entry else None


def forest_region_for_koppen(class_id, lat):
    """Map a Koppen class id to the vegetation climate-region names."""
    entry = KOPPEN_CLASSES.get(class_id)
    if not entry:
        return latitude_band_region(lat)
    code = entry[0]
    a = abs(float(lat))
    if code.startswith("A"):
        return "tropical"
    if code.startswith("B"):
        if code.endswith("h") or a < 25:
            return "subtropical"
        if a < 35:
            return "northsouth"
        if a < 55:
            return "northmiddle"
        return "northnorth"
    if code.startswith("C"):
        if a < 25:
            return "subtropical"
        if a < 35 or code[2:3] == "a":
            return "northsouth"
        return "northmiddle"
    if code.startswith("D"):
        if a >= 55 or code[2:3] in {"c", "d"}:
            return "northnorth"
        return "northmiddle"
    if code.startswith("E"):
        return "northnorth"
    return latitude_band_region(lat)


def forest_region_for_latlon(lat, lon):
    """Return the forest region selected from real Koppen climate data."""
    return forest_region_for_koppen(koppen_class_id(lat, lon), lat)
