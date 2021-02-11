#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:28:58 2020

@author: colin
"""
import os

br_un = os.environ.get("BR_GMAIL_USER")
br_pw = os.environ.get("BR_GMAIL_PASS")

print(br_un, br_pw)