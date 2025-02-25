import streamlit as st
import preprocessor, helper
import sentiment_analysis, network_analysis
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
   
   # Prepare userlist for selection and let user select user
    user_list = ["Overall"] + sorted(set(df['user']) - {'group_notification'})  
    selected_user = st.sidebar.selectbox("Select user", user_list)

    # Chat Analysis
    st.sidebar.write("## Chat Analysis")
    if st.sidebar.button("Show Analysis"):
        # Fetch Stats
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        # Display Stats
        st.title("Top Statistics")
        cols = st.columns(4)  
        stats = ["Total Messages", "Total Words", "Media Shared", "Links Shared"]  
        values = [num_messages, words, num_media_messages, num_links]  

        for col, stat, value in zip(cols, stats, values):  
            with col:  
                st.header(stat)  
                st.title(value)  

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)
        fig,ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")
        col1,col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df["count"].head(),labels=emoji_df["count"].head(),autopct="%0.2f")
            st.pyplot(fig)


    # Sentiment Analysis
    st.sidebar.write("## Sentiment Analysis")
    if st.sidebar.button("Analyze Sentiment"):
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        df = sentiment_analysis.apply_sentiment_analysis(df)
        st.title("Sentiment Analysis")
        st.write(df[["date", "user", "message", "sentiment"]])  # Display results

        st.subheader("📊 Sentiment Distribution")
        sentiment_analysis.visualize_sentiment_distribution(df)

        st.subheader("📈 Sentiment Trend Over Time")
        sentiment_analysis.sentiment_over_time(df, st)

    # Network Analysis is done when Overall is selected
    if selected_user == 'Overall':
        # Network Analysis
        st.sidebar.write("## Network Analysis")
        if st.sidebar.button("Generate Interaction Network"):
            G = network_analysis.build_interaction_network(df)
            st.title("Network Analysis")
            network_analysis.visualize_network(G, st)
            network_analysis.analyze_network(G, st)









