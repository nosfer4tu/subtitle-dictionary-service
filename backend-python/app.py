from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import requests
import psycopg2
import psycopg2.extras
import os
from contextlib import contextmanager


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

TMDB_API_KEY= os.environ.get("TMDB_API_KEY")
TMDB_SEARCH_URL = os.environ.get("TMDB_SEARCH_URL")
TMDB_IMAGE_BASE_URL = os.environ.get("TMDB_IMAGE_BASE_URL")




