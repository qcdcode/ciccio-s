#!/bin/bash

awk '{if($1=="#Bookmark"){a=!a;print $0}}(substr($1,0,1)!="#" && a==1)'
