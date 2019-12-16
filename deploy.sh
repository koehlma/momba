#!/bin/bash
echo Preparing...
mkdir ~/public/new
cp -R docs/* ~/public/new/
echo Swapping...
mv ~/public/$1 ~/public/old
mv ~/public/new ~/public/$1
echo Cleaning up...
rm -rf ~/public/old
echo Done.
