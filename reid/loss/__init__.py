from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss
from .center_loss import CenterLoss
from .matchLoss import lossMMD
from .neighbour_loss import NeiLoss
from .virtual_ce import VirtualCE, VirtualKCE

__all__ = [
    'oim', 'OIM', 'OIMLoss','NeiLoss',
    'TripletLoss', 'CenterLoss', 'lossMMD', 'VirtualCE', 'VirtualKCE'
]
