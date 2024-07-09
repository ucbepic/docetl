import streamlit as st
import sqlite3
import os
from collections import Counter
import pickle
import altair as alt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


# Path to the SQLite database inside the Modal volume
db_path = "/my_vol/chronicling_america.db"
pickle_path = "/my_vol/intermediates/entity_events.pkl"
embeddings_path = "/my_vol/intermediates/embeddings.pkl"


# Function to load entity events from the pickle file
@st.cache_data
def load_entity_events(all_entities):
    with open(pickle_path, "rb") as f:
        entity_events = pickle.load(f)

    # Filter out entities that are not in the database
    entity_events = {
        entity: events
        for entity, events in entity_events.items()
        if entity in all_entities
    }

    return entity_events


@st.cache_data
def get_all_entities():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT entity
        FROM entity_timelines
    """
    )
    entities = cursor.fetchall()
    conn.close()
    return [entity[0] for entity in entities]


@st.cache_data
def load_timeline(entity):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT entity, timeline
        FROM entity_timelines
        WHERE entity = ?
    """,
        (entity,),
    )
    data = cursor.fetchall()
    conn.close()
    return data[0][1]


# Function to load summaries for a specific entity
@st.cache_data
def load_summaries(entity):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.date, s.summary, i.pdf_file
        FROM summaries s
        JOIN issues i ON s.url = i.url
        WHERE s.summary LIKE ?
    """,
        ("%" + entity + "%",),
    )
    data = cursor.fetchall()
    conn.close()
    return data


# Function to load embeddings from the pickle file
@st.cache_data
def load_embeddings():
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


# Streamlit app
def main():
    st.title("Understanding the Chicago Tribune from Chronicling America")

    # Load all entities
    all_entities = get_all_entities()

    # Load the entity events data
    entity_events = load_entity_events(all_entities)

    # Load the embeddings
    embeddings = load_embeddings()

    if entity_events:
        # Create tabs
        tab1, tab2 = st.tabs(["Entity Timelines", "Entity Embeddings Visualization"])

        with tab1:
            # Count events for each entity
            entities_counter = Counter(
                {entity: len(events) for entity, events in entity_events.items()}
            )

            # Order entities by count in descending order
            ordered_entities = sorted(
                entities_counter.items(), key=lambda x: x[1], reverse=True
            )

            # Create a list of entities with counts
            entities_with_counts = [
                f"{entity} ({count})" for entity, count in ordered_entities
            ]

            # Sidebar dropdown to select an entity
            selected_entity_with_count = st.sidebar.selectbox(
                "Select an entity", entities_with_counts
            )
            selected_entity = selected_entity_with_count.split(" (")[0]

            if selected_entity:
                # Display the timeline / report
                st.header(f"Report for {selected_entity}")

                report = load_timeline(selected_entity)
                st.write(report)

                # Display the events
                events = entity_events[selected_entity]

                st.header(f"{len(events)} Events for {selected_entity}")

                for event in events:
                    date, description = event
                    url = f"https://chroniclingamerica.loc.gov/lccn/sn84031492/{date.replace('-', '')}/ed-1/seq-1/"
                    st.markdown(f"- [{date}]({url}): {description}")

        with tab2:
            # Visualize embeddings using Altair
            st.header("Entity Embeddings Visualization")

            # Prepare data for visualization
            entity_names = list(embeddings.keys())
            embedding_vectors = np.array(list(embeddings.values()))

            # Reduce dimensionality for visualization
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embedding_vectors)

            df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
            df["entity"] = entity_names

            point_selector = alt.selection_point("point_selection")
            interval_selector = alt.selection_interval("interval_selection")
            chart = (
                alt.Chart(df)
                .mark_circle()
                .encode(
                    x="x",
                    y="y",
                    size=alt.value(60),
                    color=alt.value("steelblue"),
                    tooltip=["entity"],
                    fillOpacity=alt.condition(
                        point_selector, alt.value(1), alt.value(0.3)
                    ),
                )
                .add_params(point_selector, interval_selector)
            )

            st.altair_chart(chart, use_container_width=True)

    else:
        st.error("No data found in the database.")


if __name__ == "__main__":
    main()
