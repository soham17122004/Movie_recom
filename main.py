import os
import pickle
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# ENVIRONMENT
# =====================================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"

TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"


if not TMDB_API_KEY:

    raise RuntimeError(
        "TMDB_API_KEY is missing. "
        "Add TMDB_API_KEY=your_api_key in .env"
    )


# =====================================
# FASTAPI
# =====================================

app = FastAPI(
    title="MovieFlix Recommendation API",
    version="1.0"
)


app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# =====================================
# FILE PATHS
# =====================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DF_PATH = os.path.join(
    BASE_DIR,
    "df.pkl"
)


INDICES_PATH = os.path.join(
    BASE_DIR,
    "indices.pkl"
)


# Your saved files currently have
# opposite names.

TFIDF_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "tfidf.pkl"
)


TFIDF_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_matrix.pkl"
)


# =====================================
# GLOBAL VARIABLES
# =====================================

df = None

indices = None

tfidf_matrix = None

tfidf_vectorizer = None

title_to_index = {}


# =====================================
# HELPER FUNCTIONS
# =====================================

def normalize_title(title):

    return str(
        title
    ).strip().lower()



def image_url(path):

    if not path:

        return None

    return (
        TMDB_IMAGE_URL
        +
        path
    )



async def tmdb_request(
    endpoint,
    parameters=None
):

    if parameters is None:

        parameters = {}


    parameters["api_key"] = (
        TMDB_API_KEY
    )


    try:

        async with httpx.AsyncClient(
            timeout=30
        ) as client:

            response = await client.get(

                TMDB_BASE_URL
                +
                endpoint,

                params=parameters

            )


    except Exception as error:

        raise HTTPException(

            status_code=502,

            detail=(
                "Could not connect "
                "to TMDB: "
                +
                str(error)
            )

        )


    if response.status_code != 200:

        raise HTTPException(

            status_code=502,

            detail=response.text

        )


    return response.json()



def create_movie_card(movie):

    return {

        "tmdb_id":
        movie.get("id"),

        "title":
        movie.get("title")
        or
        "Unknown",

        "poster_url":
        image_url(
            movie.get(
                "poster_path"
            )
        ),

        "release_date":
        movie.get(
            "release_date"
        ),

        "vote_average":
        movie.get(
            "vote_average"
        )

    }



async def search_first_movie(
    movie_title
):

    data = await tmdb_request(

        "/search/movie",

        {

            "query":
            movie_title,

            "language":
            "en-US",

            "include_adult":
            "false"

        }

    )


    results = data.get(
        "results",
        []
    )


    if not results:

        return None


    return results[0]



async def movie_details(
    tmdb_id
):

    movie = await tmdb_request(

        f"/movie/{tmdb_id}",

        {

            "language":
            "en-US"

        }

    )


    return {

        "tmdb_id":
        movie["id"],

        "title":
        movie.get(
            "title"
        ),

        "overview":
        movie.get(
            "overview"
        ),

        "release_date":
        movie.get(
            "release_date"
        ),

        "poster_url":
        image_url(
            movie.get(
                "poster_path"
            )
        ),

        "backdrop_url":
        image_url(
            movie.get(
                "backdrop_path"
            )
        ),

        "genres":
        movie.get(
            "genres",
            []
        )

    }


# =====================================
# LOAD MODEL
# =====================================

@app.on_event("startup")

def load_model():

    global df

    global indices

    global tfidf_matrix

    global tfidf_vectorizer

    global title_to_index


    files = [

        DF_PATH,

        INDICES_PATH,

        TFIDF_MATRIX_PATH,

        TFIDF_VECTORIZER_PATH

    ]


    for file_path in files:

        if not os.path.exists(
            file_path
        ):

            raise RuntimeError(

                "File missing: "

                +

                file_path

            )


    with open(
        DF_PATH,
        "rb"
    ) as file:

        df = pickle.load(
            file
        )


    with open(
        INDICES_PATH,
        "rb"
    ) as file:

        indices = pickle.load(
            file
        )


    # tfidf.pkl contains your matrix

    with open(
        TFIDF_MATRIX_PATH,
        "rb"
    ) as file:

        tfidf_matrix = (
            pickle.load(
                file
            )
        )


    # tfidf_matrix.pkl contains
    # your vectorizer

    with open(
        TFIDF_VECTORIZER_PATH,
        "rb"
    ) as file:

        tfidf_vectorizer = (
            pickle.load(
                file
            )
        )


    title_to_index = {}


    for title, index in (
        indices.items()
    ):

        try:

            if isinstance(
                index,
                pd.Series
            ):

                index = (
                    index.iloc[0]
                )


            title_to_index[

                normalize_title(
                    title
                )

            ] = int(index)


        except Exception:

            continue


    print(
        "Movie recommendation "
        "model loaded successfully"
    )


# =====================================
# FIND LOCAL MOVIE
# =====================================

def get_movie_index(
    movie_title
):

    normalized = normalize_title(
        movie_title
    )


    if normalized in title_to_index:

        return title_to_index[
            normalized
        ]


    raise HTTPException(

        status_code=404,

        detail=(

            f"'{movie_title}' "

            "was not found "

            "in the local dataset"

        )

    )


# =====================================
# RECOMMENDATION MODEL
# =====================================

def recommend_movies(
    movie_title,
    top_n=10
):

    movie_index = get_movie_index(
        movie_title
    )


    similarity = cosine_similarity(

        tfidf_matrix[
            movie_index
        ],

        tfidf_matrix

    ).flatten()


    sorted_indices = (

        np.argsort(
            similarity
        )[::-1]

    )


    recommendations = []


    for index in sorted_indices:

        index = int(index)


        if index == movie_index:

            continue


        if index >= len(df):

            continue


        title = str(

            df.iloc[
                index
            ]["title"]

        )


        recommendations.append(

            {

                "title":
                title,

                "score":
                round(

                    float(

                        similarity[
                            index
                        ]

                    ),

                    4

                )

            }

        )


        if len(
            recommendations
        ) >= top_n:

            break


    return recommendations


# =====================================
# HEALTH
# =====================================

@app.get("/")

def root():

    return {

        "message":

        "MovieFlix API is running"

    }



@app.get("/health")

def health():

    return {

        "status":

        "ok"

    }


# =====================================
# HOME MOVIES
# =====================================

@app.get("/home")

async def home(

    category: str = "popular",

    limit: int = 20

):


    if category == "trending":

        endpoint = (

            "/trending/movie/day"

        )

    else:

        allowed = [

            "popular",

            "top_rated",

            "upcoming",

            "now_playing"

        ]


        if category not in allowed:

            category = "popular"


        endpoint = (

            f"/movie/{category}"

        )


    data = await tmdb_request(

        endpoint,

        {

            "language":
            "en-US",

            "page":
            1

        }

    )


    movies = []


    for movie in (

        data.get(
            "results",
            []
        )[:limit]

    ):

        movies.append(

            create_movie_card(
                movie
            )

        )


    return movies


# =====================================
# TMDB SEARCH
# =====================================

@app.get("/tmdb/search")

async def tmdb_search(

    query: str

):


    return await tmdb_request(

        "/search/movie",

        {

            "query":
            query,

            "language":
            "en-US"

        }

    )


# =====================================
# MOVIE DETAILS
# =====================================

@app.get(
    "/movie/id/{tmdb_id}"
)

async def get_details(
    tmdb_id: int
):

    return await movie_details(
        tmdb_id
    )


# =====================================
# TF-IDF API
# =====================================

@app.get(
    "/recommend/tfidf"
)

def recommend_tfidf(

    title: str,

    top_n: int = 10

):

    return recommend_movies(

        title,

        top_n

    )


# =====================================
# GENRE RECOMMENDATION
# =====================================

@app.get(
    "/recommend/genre"
)

async def genre_recommendation(

    tmdb_id: int,

    limit: int = 18

):


    details = await movie_details(
        tmdb_id
    )


    genres = details.get(
        "genres",
        []
    )


    if not genres:

        return []


    genre_id = genres[0]["id"]


    data = await tmdb_request(

        "/discover/movie",

        {

            "with_genres":
            genre_id,

            "sort_by":
            "popularity.desc",

            "language":
            "en-US"

        }

    )


    cards = []


    for movie in data.get(

        "results",

        []

    ):


        if (

            movie.get("id")

            ==

            tmdb_id

        ):

            continue


        cards.append(

            create_movie_card(
                movie
            )

        )


        if len(cards) >= limit:

            break


    return cards


# =====================================
# COMPLETE SEARCH
# =====================================

@app.get("/movie/search")

async def complete_search(

    query: str,

    tfidf_top_n: int = 12,

    genre_limit: int = 12

):


    selected_movie = (

        await search_first_movie(
            query
        )

    )


    if not selected_movie:

        raise HTTPException(

            status_code=404,

            detail=(

                "Movie was "

                "not found"

            )

        )


    details = await movie_details(

        selected_movie["id"]

    )


    recommendations = []


    try:

        local_movies = (

            recommend_movies(

                details["title"],

                tfidf_top_n

            )

        )


    except Exception:

        try:

            local_movies = (

                recommend_movies(

                    query,

                    tfidf_top_n

                )

            )


        except Exception:

            local_movies = []


    for recommendation in (

        local_movies

    ):


        tmdb_movie = (

            await search_first_movie(

                recommendation[
                    "title"
                ]

            )

        )


        card = None


        if tmdb_movie:

            card = (

                create_movie_card(

                    tmdb_movie

                )

            )


        recommendations.append(

            {

                "title":

                recommendation[
                    "title"
                ],

                "score":

                recommendation[
                    "score"
                ],

                "tmdb":

                card

            }

        )


    genre_movies = (

        await genre_recommendation(

            details[
                "tmdb_id"
            ],

            genre_limit

        )

    )


    return {

        "query":

        query,

        "movie_details":

        details,

        "tfidf_recommendations":

        recommendations,

        "genre_recommendations":

        genre_movies

    }
