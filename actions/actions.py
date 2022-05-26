# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import pickle
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from fast_autocomplete import AutoComplete

# rasa imports
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import ActionExecuted, SlotSet, FollowupAction, UserUttered

import sqlite3
import random
from fuzzywuzzy import process
import collections

import re
import json
import os
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import NavigableString

class SearchAutocomplete():
    def __init__(self, input_file, top_n=20):
        with open(input_file, 'rb') as f:
            search_obj = pickle.load(f)

        ac = AutoComplete(words={})
        ac._dwg = search_obj["acp"]
        search_obj["acp"] = ac

        self.autocomplete = search_obj['acp']
        self.df = search_obj['df']
        self.df_perm = search_obj['df_perm']
        self.top_n = top_n

    def search_query(self, input_value):
        res = self.autocomplete.search(word=input_value, max_cost=20, size=self.top_n)
        res = self.unique( self.flatten(res) )
        ids = self.df_perm[self.df_perm.text.isin(res)].id.unique()
        yhat = self.df[self.df.id.isin(ids)].copy()

        # top5 based on text
        yhat_retrieval = yhat.head(5)
        return yhat_retrieval

    def unique(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def flatten(self, t):
        """Convert a list of lists in one flattened list."""
        return [item for sublist in t for item in sublist]


def get_last_flow_intent(tracker_events):
    # define intents we want
    keep_intents = ["howto", "commonqueries"]

    user_events = [x for x in tracker_events if x['event'] == 'user']
    previous_intents = [x["parse_data"]['intent']['name'] for x in user_events]
    flow_intents = [x for x in previous_intents if x in keep_intents]
    last_flow_intent = flow_intents[-1]
    return last_flow_intent

# read ACP object for how to guides and common queries
search_howto = SearchAutocomplete(input_file="objects/acp_howto.pkl", top_n=20)
search_queries = SearchAutocomplete(input_file="objects/acp_queries.pkl", top_n=20)


class ActionQueries(Action):
    def name(self) -> Text:
        return "action_queries"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # figure out the
        last_flow_intent = get_last_flow_intent(tracker.events)
        #print(tracker.events)
        print("Last intent: ", last_flow_intent)

        # get input text
        input_query = tracker.latest_message['text']

        if last_flow_intent == "commonqueries":
            # how to guides retrieval
            df = search_queries.search_query(input_query)

            if df.shape[0] > 0:
                df_top = df.head(1)
                answer_text = df_top['answer'].to_list()[0]
                query_text = df_top['query'].to_list()[0].title()
                output_text = f'To diagnose or troubleshoot "{query_text}", here are some suggestions:\n' + answer_text
            else:
                output_text = "I don't have an answer at the moment but please check out our [articles](https://shop.advanceautoparts.com/r/search?f%5B0%5D=subcategory%3A331)!"
        else:
            # Curated data retrieval
            df = search_howto.search_query(input_query)
            base_url = 'https://shop.advanceautoparts.com'
            formatted_links = []
            for inx, row in df.iterrows():
                url = base_url + row['link']
                link = f"- [{row['title'].title()}]({url})"
                formatted_links.append(link)

            formatted_links = "\n".join(formatted_links)
            if len(formatted_links) > 0:
                output_text = "Check out these links:\n" + formatted_links
            else:
                output_text = "I don't have an answer at the moment but please check out our [articles](https://shop.advanceautoparts.com/r/search?f%5B0%5D=subcategory%3A56)!"

        # output whatever text is available
        dispatcher.utter_message(text=output_text)

        return []



########## Driverside classes ##############
class ActionQueryResource(Action):

    def name(self) -> Text:
        return "action_query_resource"

    def run(self, dispatcher: "CollectingDispatcher", tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Runs a query using answer id and finds the next question.
        The possible answers are returned as buttons for the user to select from.
        """
        conn = DbQueryingMethods.create_connection(db_file="./actions/resourcesDB")
        conn.row_factory = sqlite3.Row
        # get next question from the
        answer_id = int(tracker.get_slot("answer_id"))
        print(answer_id)

        apology = "I couldn't find exactly what you wanted, but you might like this."
        return_text = ""
        if answer_id == None:
            return_text = apology
            print(apology)

        question = DbQueryingMethods.find_next_ques(conn, answer_id)
        if question is not None:
            print(f"question found {question}")
            possible_answers = DbQueryingMethods.fetch_answer(conn,
                question['ques_id'])
            buttons = []
            if possible_answers is not None:
                # if the number of possible answers are more, return them as options
                if len(possible_answers) > 1:
                    buttons = [
                                {
                                    'title': possible_answer['ans'],
                                    'payload':"/inform{\"answer_id\": \"" + str(possible_answer["ans_id"]) + "\"}"
                                }
                            for possible_answer in possible_answers
                            ]
                    if answer_id == 0:
                        dispatcher.utter_message(
                            text=question['ques'] + "\nYou can also ask question by directly typing the query.", buttons=buttons)
                        return [SlotSet("continue_loop", True)]
                    else:
                        dispatcher.utter_message(
                            text=question['ques'], buttons=buttons)
                        return [SlotSet("continue_loop", True)]
                # else return it as text
                else:
                    dispatcher.utter_message(text= question['ques'] + "\n" + possible_answers[0]['ans']) 
                    return  [SlotSet("continue_loop", False), FollowupAction("utter_intro")]
        else:
            dispatcher.utter_message(text= apology)

        return [SlotSet("continue_loop", False), FollowupAction("utter_intro")]


class ActionSetProblemSlot(Action):

    def name(self) -> Text:
        return "action_set_problem_slot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [
            SlotSet("problem_type", tracker.get_intent_of_latest_message()),
            SlotSet("continue_loop",True),
            SlotSet("answer_id",0)
        ]


class DbQueryingMethods:
    def create_connection(db_file):
        """
        create a database connection to the SQLite database
        specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
        except Exception as e:
            print(e)

        return conn

    def get_closest_value(conn, slot_name, slot_value):
        """ Given a database column & text input, find the closest
        match for the input in the column.
        """
        # get a list of all distinct values from our target column
        fuzzy_match_cur = conn.cursor()
        fuzzy_match_cur.execute(f"""SELECT DISTINCT {slot_name}
                                FROM answer""")
        column_values = fuzzy_match_cur.fetchall()

        top_match = process.extractOne(slot_value, column_values)

        return(top_match[0])

    def rows_info_as_text(rows):
        """
        Return one of the rows (randomly selcted) passed in
        as a human-readable text. If there are no rows, returns
        text to that effect.
        """
        if len(list(rows)) < 1:
            return "There are no resources matching your query."
        else:
            for row in random.sample(rows, 1):
                return f"Try the {(row[4]).lower()} {row[0]} by {row[1]}. You can find it at {row[2]}."

    def find_next_ques(conn, ans_id):
        """
        Returns the next question from the question table based on the answer selected by the user
        """
        cur = conn.cursor()
        if ans_id> -1:
            query = f"""
                Select * from question where ques_id = {ans_id}
            """
            cur.execute(query)
            rows = cur.fetchall()
            cols = [x[0] for x in cur.description]
            if len(rows) > 0:
                return {
                    k:v for k,v in zip(cols, rows[0])
                }
        return None

    def fetch_answer(conn, ques_id):
        """
        Returns all the answers and their ids which might be linked to a question
        """
        if ques_id > -1:

            cur = conn.cursor()

            query = f"""
            Select * from answer where ques_id = {ques_id}
            """
            cur.execute(query)
            rows = cur.fetchall()
            return rows
        return None

    def close_connection(conn):
        """
        closes the connection to the sqlite3 database file
        """
        if conn is not None:
            conn.close()
        return
