import os

from config import log_prefix, model_prefix, feature_prefix, clean_input_prefix

if not os.path.exists(clean_input_prefix):
    os.makedirs(clean_input_prefix)
if not os.path.exists(feature_prefix)
    os.makedirs(feature_prefix)
if not os.path.exists('%s/cv'%feature_prefix):
    os.makedirs('%s/cv'%feature_prefix)
if not os.path.exists('%s/all'%feature_prefix):
    os.makedirs('%s/all'%feature_prefix)
if not os.path.exists('%s/cv'%model_prefix):
    os.makedirs('%s/cv'%model_prefix)
if not os.path.exists('%s/all'%model_prefix):
    os.makedirs('%s/all'%model_prefix)
if not os.path.exists(log_prefix):
    os.makedirs(log_prefix)
