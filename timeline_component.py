import streamlit as st

def render_timeline(timeline_data):
    st.subheader("📅 Timeline View")
    people = sorted({p for event in timeline_data for p in event["AssociatedPeople"]})
    themes = sorted({t for event in timeline_data for t in event["ThematicCategories"]})

    selected_people = st.multiselect("Filter by People", people)
    selected_themes = st.multiselect("Filter by Themes", themes)

    for event in timeline_data:
        if selected_people and not set(event["AssociatedPeople"]).intersection(selected_people):
            continue
        if selected_themes and not set(event["ThematicCategories"]).intersection(selected_themes):
            continue

        st.markdown(f"### {event['Summary']}")
        st.markdown(f"> {event['FormattedSnippet']}")
        st.caption(f"{event['StandardizedDate']} – {', '.join(event['AssociatedPlaces'])} – Source: {event['SourceDocument']} (Page {event['SourcePage']})")
