import os
import json

from django.shortcuts import render

def index(request):
    """View function for home page of site."""

    test_result = "/home/roigcarlo/KratosAI/NACA/results/dataset_02_10_2023_1_xyz_2804537.csv"

    with open(test_result, "r") as json_file:
        data = json.load(json_file) 

    context = {
        'list_of_files': data["data"],
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'index.html', context=context)