import requests
import streamlit as st


# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(

    page_title="MovieFlix",

    page_icon="🎬",

    layout="wide"

)


API_URL = "http://127.0.0.1:8000"


# =====================================
# CSS
# =====================================

st.markdown(

"""

<style>

.stApp {

background-color: #090909;

color: white;

}


.main-title {

font-size: 52px;

font-weight: bold;

color: #e50914;

}


.movie-title {

font-size: 16px;

font-weight: bold;

color: white;

min-height: 45px;

}


.rating {

color: #ffd700;

font-weight: bold;

}


.stButton button {

background-color: #e50914;

color: white;

border-radius: 8px;

border: none;

}

</style>

""",

unsafe_allow_html=True

)


# =====================================
# API FUNCTION
# =====================================

def api_request(

    endpoint,

    parameters=None

):


    try:

        response = requests.get(

            API_URL
            +
            endpoint,

            params=parameters,

            timeout=60

        )


        if response.status_code == 200:

            return response.json()


        st.error(

            f"Backend error: "

            f"{response.status_code}"

        )


    except Exception as error:

        st.error(

            "Backend is not running. "

            +

            str(error)

        )


    return None


# =====================================
# MOVIE CARD
# =====================================

def movie_cards(

    movies,

    number_of_columns=5

):


    if not movies:

        st.warning(

            "No movies found."

        )

        return


    columns = st.columns(

        number_of_columns

    )


    for index, movie in (

        enumerate(movies)

    ):


        with columns[

            index

            %

            number_of_columns

        ]:


            poster = movie.get(

                "poster_url"

            )


            if poster:

                st.image(

                    poster,

                    use_container_width=True

                )


            else:

                st.info(

                    "No poster"

                )


            st.markdown(

                f"""

                <p class="movie-title">

                {

                movie.get(

                    "title",

                    "Unknown"

                )

                }

                </p>

                """,

                unsafe_allow_html=True

            )


            rating = movie.get(

                "vote_average"

            )


            if rating is not None:

                st.markdown(

                    f"""

                    <p class="rating">

                    ⭐ {rating:.1f}/10

                    </p>

                    """,

                    unsafe_allow_html=True

                )


            release_date = (

                movie.get(

                    "release_date"

                )

            )


            if release_date:

                st.caption(

                    "📅 "

                    +

                    release_date[:4]

                )


# =====================================
# HEADER
# =====================================

st.markdown(

"""

<p class="main-title">

🎬 MovieFlix

</p>

""",

unsafe_allow_html=True

)


st.write(

"Discover movies and get "

"AI-powered recommendations."

)


# =====================================
# SEARCH
# =====================================

movie_name = st.text_input(

"Search your favourite movie",

placeholder=(

"Avatar, Batman, Avengers..."

)

)


search = st.button(

"🔍 Search Movie",

use_container_width=True

)


# =====================================
# SEARCH RESULT
# =====================================

if search and movie_name.strip():


    with st.spinner(

        "Finding movies..."

    ):


        result = api_request(

            "/movie/search",

            {

                "query":

                movie_name.strip(),

                "tfidf_top_n":

                12,

                "genre_limit":

                12

            }

        )


    if result:


        details = result[

            "movie_details"

        ]


        st.divider()


        poster_column, information = (

            st.columns(

                [1, 3]

            )

        )


        with poster_column:


            if details.get(

                "poster_url"

            ):


                st.image(

                    details[

                        "poster_url"

                    ],

                    use_container_width=True

                )


        with information:


            st.title(

                details.get(

                    "title",

                    ""

                )

            )


            st.write(

                "📅 Release:",

                details.get(

                    "release_date",

                    "-"

                )

            )


            genres = [


                genre["name"]


                for genre in

                details.get(

                    "genres",

                    []

                )

            ]


            st.write(

                "🎭 Genres:",

                ", ".join(

                    genres

                )

            )


            st.subheader(

                "Overview"

            )


            st.write(

                details.get(

                    "overview"

                )

                or

                "No overview available."

            )


        # TF-IDF RECOMMENDATIONS

        st.divider()


        st.header(

            "🤖 Recommended For You"

        )


        recommendation_cards = []


        for movie in result.get(

            "tfidf_recommendations",

            []

        ):


            card = movie.get(

                "tmdb"

            )


            if card:

                recommendation_cards.append(

                    card

                )


        movie_cards(

            recommendation_cards,

            5

        )


        # GENRE MOVIES

        st.divider()


        st.header(

            "🎭 More Movies Like This"

        )


        movie_cards(

            result.get(

                "genre_recommendations",

                []

            ),

            5

        )


# =====================================
# HOME
# =====================================

else:


    category = st.selectbox(

        "Choose movie category",

        [

            "popular",

            "trending",

            "top_rated",

            "upcoming",

            "now_playing"

        ]

    )


    headings = {

        "popular":

        "🔥 Popular Movies",

        "trending":

        "📈 Trending Movies",

        "top_rated":

        "⭐ Top Rated Movies",

        "upcoming":

        "🎞 Upcoming Movies",

        "now_playing":

        "🍿 Now Playing"

    }


    st.header(

        headings[category]

    )


    movies = api_request(

        "/home",

        {

            "category":

            category,

            "limit":

            20

        }

    )


    movie_cards(

        movies,

        5

    )