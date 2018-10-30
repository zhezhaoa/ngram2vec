#!/bin/sh

# Cleaning English corpus. Strategy used in hyperwords toolkit.
# Please use gnu-sed on Mac by running `brew install gnu-sed --with-default-names`. 
# Note that utf-8 and ascii are identical for English corpus.
iconv -c -f utf-8 -t ascii $1 | tr '[A-Z]' '[a-z]' | sed "s/[^a-z0-9]*[ \t\n\r][^a-z0-9]*/ /g" | sed "s/[^a-z0-9]*$/ /g" | sed "s/^[^a-z0-9]*//g" | sed "s/  */ /g"
