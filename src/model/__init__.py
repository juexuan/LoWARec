from model.lowarec import LoWARec
from model.bsarec import BSARecModel
from model.caser import CaserModel
from model.gru4rec import GRU4RecModel
from model.sasrec import SASRecModel
from model.bert4rec import BERT4RecModel
from model.fmlprec import FMLPRecModel
from model.duorec import DuoRecModel
from model.fearec import FEARecModel
from model.lightsans import LightSANs


MODEL_DICT = {

    "lowarec": LoWARec,
    "lightsans": LightSANs,
    "caser": CaserModel,
    "gru4rec": GRU4RecModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
    "fmlprec": FMLPRecModel,
    "duorec": DuoRecModel,
    "fearec": FEARecModel,
    }