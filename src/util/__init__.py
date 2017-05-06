
from .experiment_utils import *
from .code_utils import *

def fill_dict(d1, d2):
     d22 = dict(d2)
     d11 = dict(d1)
     d22.update(d11)
     d11.update(d22)
     return d11

def about(x, LINE=80, SINGLE_LINE=False):
    '''
    author: Yogesh Garg (https://github.com/yogeshg)
    '''
    s ='type:'+str(type(x))+' '
    try:
        s+='shape:'+str(x.shape)+' '
    except Exception as e:
        pass
    try:
        s+='dtype:'+str(x.dtype)+' '
    except Exception as e:
        pass
    try:
        s+='size:'+str(x.shape)+' '
    except Exception as e:
        try:
            s+='len:'+str(len(x))+' '
        except Exception as e:
            pass
    try:
        s1 = str(x)
        if(SINGLE_LINE):
            s1 = ' '.join(s1.split('\n'))
            extra = (len(s)+len(s1)) - LINE
            if(extra > 0):
                s1 = s1[:-(extra+3)]+'...'
            s+=s1
        else:
            s+='\n'+s1
    except Exception as e:
        pass
    return s

