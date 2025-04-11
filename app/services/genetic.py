from fastapi import  HTTPException
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from bson import ObjectId

from app.database import users_collection, vectors_collection, jobs_collection

async def get_job_candidates(job_id: str):
    """Récupère les candidats associés à un job"""
    job = await jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        return []
    
    users = await users_collection.find({"job_id": job_id}).to_list(None)
    # return [format_user(user) for user in users]
    return [user for user in users]

async def get_job_candidates_data(job_id: str):
    """Récupère les vecteurs et les utilisateurs associés à un job donné"""
    try:
        # Assurez-vous que le job existe
        job = await jobs_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(404, "Job not found")

        # Récupérer tous les utilisateurs liés au job
        users = await users_collection.find({"job_id": job_id}).to_list(None)
        if not users:
            return {"vectors": [], "user_vector_map": []}

        # Récupérer les vecteurs pour ces utilisateurs
        user_ids = [str(user["_id"]) for user in users]
        vectors = await vectors_collection.find({"user_id": {"$in": user_ids}}).to_list(None)

        # Créer un mapping user_id -> vector
        user_vector_dict = {v["user_id"]: v["vector"] for v in vectors}

        # Construire la liste des vecteurs
        vectors_only = [user_vector_dict.get(str(user["_id"]), []) for user in users]

        # Construire le lien complet candidat + vecteur
        user_vector_map = [
            {"user": user, "vector": user_vector_dict.get(str(user["_id"]), [])}
            for user in users
        ]

        return {
            "vectors": vectors_only,
            "user_vector_map": user_vector_map
        }

    except Exception as e:
        raise HTTPException(500, f"Erreur lors de la récupération des vecteurs candidats : {str(e)}")

def get_user_by_id(user_id, collection):
    """
    Récupère un utilisateur à partir de son ID dans une collection MongoDB.

    :param user_id: str ou ObjectId - L'ID de l'utilisateur
    :param collection: Collection MongoDB (ex: users_collection)
    :return: dict ou None - L'utilisateur trouvé ou None
    """
    try:
        # Convertir en ObjectId si ce n'est pas déjà le cas
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        return collection.find_one({'_id': user_id})
    except Exception as e:
        print(f"Erreur lors de la recherche de l'utilisateur: {e}")
        return None
    
def serialize_mongo_document(doc):
    if not doc:
        return None
    doc["_id"] = str(doc["_id"])
    return doc

class CandidateModel:
    """
    Mathematical model for evaluating candidate suitability based on multiple criteria.
    """

    @staticmethod
    def education_level(x1):
        """Encodes education level as an ordinal scale."""
        return x1  # Should be encoded externally as Bachelor's=3, Master's=5, PhD=8

    @staticmethod
    def work_experience(x2, X_max):
        """Normalizes work experience."""
        return x2 / X_max if X_max > 0 else 0  # Avoid division by zero

    @staticmethod
    def skill_match(skills_acquired, skills_required):
        """Computes skill match ratio."""
        return skills_acquired / skills_required if skills_required > 0 else 0

    @staticmethod
    def assessment_test_score(x4):
        """Normalizes assessment test score."""
        return x4 / 100  # Assuming scores are between 0 and 100

    @staticmethod
    def professional_certifications(x5):
        """Counts relevant certifications."""
        return x5  # Number of relevant certifications

    @staticmethod
    def language_proficiency(x6):
        """Encodes language proficiency as binary."""
        # return 1 if x6 else 0  # Bilingual=1, Monolingual=0
        return 1 if len(x6) > 1 else 0,

    @staticmethod
    def availability(x7, lambda_):
        """Models availability as an exponential penalty function."""
        if not (0 < lambda_ <= 1):
            raise ValueError("Lambda must be between 0 and 1.")
        return np.exp(-lambda_ * x7)

    @staticmethod
    def age_suitability(x8, a_min, a_max, a_optimal, sigma):
        """Applies penalty if candidate falls outside preferred age range."""
        if a_min <= x8 <= a_max:
            return 1
        return np.exp(-abs(x8 - a_optimal) / sigma)

    @staticmethod
    def compute_total_score(candidate_data, weights):
        """Computes the total candidate score based on weighted sum of criteria."""
        return sum(w * f for w, f in zip(weights, candidate_data))
    
    
class GeneticAlgorithm:
    """
    Real-Coded Genetic Algorithm for optimizing candidate selection.
    """
    def __init__(self, job_id: str, population_size, generations, mutation_rate, crossover_rate, X_max, a_min, a_max):
        self.job_id = job_id
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.X_max = X_max  # Max experience
        self.a_min = a_min  # Min age
        self.a_max = a_max  # Max age

        # Complete traceability
        self.history = {
            'virtual_mutations': [],
            'crossovers': [],
            'final_selection': []
        }

        # Storage of initial candidates
        self.initial_population = None
        self.parent_child_mapping = {}  # To associate each parent with its children

        # New: mapping to associate a vector with a unique ID and vice versa
        self.candidate_id_map = {}   # dict:  tuple(candidate) -> candidate_id
        self.id_to_vector_map = {}   # dict:  candidate_id -> tuple(candidate)

        # Current generation (for history)
        self.current_generation = 0
    
    # async def initialize_population(self):
    #     """Récupère les candidats associés au job"""
    #     job_candidates = await get_job_candidates_data(self.job_id)
        
    #     self.full_job_candidates = job_candidates["user_vector_map"]
    #     return job_candidates["vectors"]
    
    async def initialize_population(self):
        """Récupère les candidats associés au job"""
        job_candidates = await get_job_candidates_data(self.job_id)

        user_vector_map = job_candidates["user_vector_map"]
        
        # On ne garde que les vecteurs non vides
        vectors_only = [entry["vector"] for entry in user_vector_map if entry["vector"]]


        # On stocke les mappings ID -> vecteur et vecteur -> ID pour le reste de l’algo
        self.id_to_vector_map = {
            str(entry["user"]["_id"]): entry["vector"]
            for entry in user_vector_map if entry["vector"]
        }
        self.candidate_id_map = {
            tuple(entry["vector"]): str(entry["user"]["_id"])
            for entry in user_vector_map if entry["vector"]
        }
        self.id_to_user_map = {
            str(entry["user"]["_id"]): entry["user"]
            for entry in user_vector_map if entry["vector"]
        }
        
        self.vector_to_user = {
            tuple(entry["vector"]): entry["user"]
            for entry in user_vector_map if entry["vector"]
        }

        # tous les utilisateurs avec leur vecteur pour référence
        self.full_job_candidates = user_vector_map

        return  vectors_only

    def fitness_function(self, candidate):
        """Evaluates the quality (fitness) of a candidate."""
        score = CandidateModel.compute_total_score(candidate, self.weights)
        if score is None:
            raise ValueError(f"The score is none for the candidat : {candidate}")
        return score

    def mutate(self, candidate):
        """Mutation that respects the constraints of each variable. (Improved mutation)"""
        if np.random.rand() < self.mutation_rate:
            mutation = candidate.copy()
            mutation[0] = random.randint(1, 8)  # x1: Education category
            mutation[1] = random.randint(1, self.X_max)  # x2: Experience
            mutation[2] = np.clip(mutation[2] + np.random.normal(0, 0.1), 0, 1)  # x3: Skill
            mutation[3] = np.clip(mutation[3] + np.random.normal(0, 5), 0, 100)  # x4: Test score
            mutation[4] = random.randint(0, 10)  # x5: Certifications
            mutation[5] = random.choice([0, 1])  # x6: Language
            mutation[6] = random.randint(1, 52)  # x7: Availability
            mutation[7] = random.randint(self.a_min, self.a_max)  # x8: Age
            mutation[8] = np.clip(mutation[8] + np.random.normal(0, 0.2), 1, 5)  # x9: Qualitative
            mutation[9] = np.clip(mutation[9] + np.random.normal(0, 0.1), 0, 1)  # x10: Factor
            mutation[10] = np.clip(mutation[10] + np.random.normal(0, 0.1), 0, 1)  # x11: Other factor
            return mutation
        return candidate
        
    def virtual_targeted_mutation(self, candidate, boost_factor=1.2):
        """
        Performs a virtual targeted mutation on a candidate by boosting a random criterion.
        """
        candidate_virtual = candidate.copy()

        # Randomly select an index between 0 and 6 (inclusive)
        index_to_boost = random.choice([0, 1, 2, 3, 4, 5, 6])

        bounds = [
            (1, 8), (1, self.X_max), (0, 1), (0, 100),
            (0, 10), (0, 1), (1, 52),
            (self.a_min, self.a_max), (1, 5), (0, 1), (0, 1)
        ]

        if isinstance(candidate_virtual[index_to_boost], (int, float)):
            original_value = candidate_virtual[index_to_boost]
            boosted_value = original_value * boost_factor
            min_val, max_val = bounds[index_to_boost]

            candidate_virtual[index_to_boost] = np.clip(boosted_value, min_val, max_val)

        virtual_score = self.fitness_function(candidate_virtual)
        original_score = self.fitness_function(candidate)

        improved = virtual_score > original_score

        # Track in history with boosted index
        self.history['virtual_mutations'].append({
            'generation': self.current_generation,
            'candidate_vector': candidate,
            'boosted_candidate_vector': candidate_virtual,
            'boosted_index': index_to_boost,
            'improved': improved,
            'original_score': original_score,
            'virtual_score': virtual_score
        })

        return improved, virtual_score

    def crossover(self, parent1, parent2):
        """BLX-alpha crossover with bound and type restriction. """
        child1, child2 = parent1.copy(), parent2.copy()
        alpha = 0.5

        # Definition of bounds for each variable
        bounds = [
            (1, 8),                        # x1: Education (1 to 8) (integer)
            (1, self.X_max),               # x2: Experience (1 to Xmax) (integer)
            (0.0, 1.0),                     # x3: Skill (0 to 1) (float)
            (0.0, 100.0),                   # x4: Test score (0 to 100) (float)
            (0, 10),                        # x5: Certifications (0 to 10) (integer)
            (0, 1),                         # x6: Language (0 or 1) (integer)
            (1, 52),                        # x7: Availability (1 to 52 weeks) (integer)
            (self.a_min, self.a_max),       # x8: Age (bounded) (integer)
            (1.0, 5.0),                     # x9: Qualitative score (1 to 5) (float)
            (0.0, 1.0),                     # x10: Factor (0 to 1) (float)
            (0.0, 1.0)                      # x11: Other factor (0 to 1) (float)
        ]

        for i in range(len(parent1)):
            min_val, max_val = bounds[i]
            if isinstance(parent1[i], int):  # Handle integers with partial exchange
                if np.random.rand() < 0.5 :
                    child1[i], child2[i] = parent2[i], parent1[i]
            else:  # Handle floats with BLX-alpha
                range_val = max_val - min_val
                child1[i] = np.clip(
                    np.random.uniform(min(parent1[i], parent2[i]) - alpha * range_val,
                                    max(parent1[i], parent2[i]) + alpha * range_val),
                    min_val, max_val
                )
                child2[i] = np.clip(
                    np.random.uniform(min(parent1[i], parent2[i]) - alpha * range_val,
                                    max(parent1[i], parent2[i]) + alpha * range_val),
                    min_val, max_val
                )

        return child1, child2 

    def crossover_collaboration(self, parent1, parent2):
        """
        Collaboration between two parents: synergy is tested.
        If the average score is better than the sum of individual scores,
        the parents are rewarded.
        """
        child1, child2 = self.crossover(parent1, parent2)

        parent_scores = [self.fitness_function(parent1), self.fitness_function(parent2)]
        children_scores = [self.fitness_function(child1), self.fitness_function(child2)]

        synergy = np.mean(children_scores) > np.mean(parent_scores)

        self.history['crossovers'].append({
            'generation': self.current_generation,
            'parent1_vector': parent1,
            'parent2_vector': parent2,
            'child1_vector': child1,
            'child2_vector': child2,
            'parent1_score': parent_scores[0],
            'parent2_score': parent_scores[1],
            'child1_score': children_scores[0],
            'child2_score': children_scores[1],
            'synergy': synergy
        })

        # Reward: a virtual bonus is added if the collaboration is successful
        reward_parent1 = parent_scores[0] * 1.05 if synergy else parent_scores[0]
        reward_parent2 = parent_scores[1] * 1.05 if synergy else parent_scores[1]

        return child1, child2, reward_parent1, reward_parent2

    def select_parents(self, population, fitness):
        """Tournament selection of parents."""
        selected = []
        for _ in range(len(population)):
            i, j = np.random.randint(0, len(population), 2)
            selected.append(population[i] if fitness[i] > fitness[j] else population[j])
        return np.array(selected)    

    def final_selection(self, population, fitness, mutation_flags, crossover_flags):
        """
        Final selection with consideration of mutation and collaboration bonuses.
        The candidate ID (candidate_id) is now recorded in the history
        instead of the complete vector.
        """
        bonus_mutation = 0.05
        bonus_collaboration = 0.05

        final_scores = []
        for idx, candidate in enumerate(population):
            score = fitness[idx]
            bonus = 0
            if mutation_flags[idx]:
                bonus += bonus_mutation
            if crossover_flags[idx]:
                bonus += bonus_collaboration
            final_score = score * (1 + bonus)
            final_scores.append(final_score)

            # Retrieve the candidate ID
            # cand_key = tuple(candidate)
            cand_key = tuple(np.array(candidate).flatten())
            # If the candidate is new (child not yet seen), a new ID can be created
            if cand_key not in self.candidate_id_map:
                new_id = len(self.candidate_id_map)  # Next ID
                self.candidate_id_map[cand_key] = new_id
                self.id_to_vector_map[new_id] = cand_key

            candidate_id = self.candidate_id_map[cand_key]

            self.history['final_selection'].append({
                'generation': self.current_generation,
                'candidate_id': candidate_id,
                'raw_score': score,
                'bonus': bonus,
                'final_score': final_score,
                'operation': 'final_selection',
                'description': "Final selection with mutation and collaboration bonuses"
            })

        # Select the best based on final scores
        selected_indices = np.argsort(final_scores)[::-1]
        selected_population = population[selected_indices]

        return selected_population

    async def run(self, weights):
        """Main loop of the GA with virtual targeted mutation and collaborative crossover."""
        
        self.weights = weights
        # print("candidates: ", population)
        population = await self.initialize_population()
        # print("Appropriates candidates:", population[1:])
        population = [vector_c  for vector_c in population if vector_c]
        
        for generation in range(self.generations):
            self.current_generation = generation
            # print(f"\n=== Generation {generation + 1} ===")
            fitness = np.array([self.fitness_function(candidate) for candidate in population])

            # Virtual targeted mutation: identify those that can be "improved"
            mutation_flags = []
            virtual_scores = []
            for candidate in population:
                
                improved, virtual_score = self.virtual_targeted_mutation(candidate)
                mutation_flags.append(improved)
                virtual_scores.append(virtual_score)

            # Parent selection
            parents = self.select_parents(population, fitness)

            # Collaborative crossover with synergy
            offspring = []
            crossover_flags = []

            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1 if i+1 < len(parents) else 0]

                child1, child2, reward1, reward2 = self.crossover_collaboration(parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)

                # Mark if they received a bonus
                crossover_flags.append(reward1 > self.fitness_function(parent1))
                crossover_flags.append(reward2 > self.fitness_function(parent2))

                # Associate children with their parents
                # self.parent_child_mapping.setdefault(tuple(parent1), []).append(child1)
                # self.parent_child_mapping.setdefault(tuple(parent2), []).append(child2)
                self.parent_child_mapping.setdefault(tuple(np.array(parent1).flatten()), []).append(child1)
                self.parent_child_mapping.setdefault(tuple(np.array(parent2).flatten()), []).append(child2)


            offspring = np.array([self.mutate(child) for child in offspring])

            # Final selection with bonuses
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.array([self.fitness_function(c) for c in combined_population])

            combined_mutation_flags = mutation_flags + [False] * len(offspring)
            combined_crossover_flags = crossover_flags + [False] * len(population)

            population_after = self.final_selection(
                combined_population,
                combined_fitness,
                combined_mutation_flags,
                combined_crossover_flags
            )[:self.population_size]

        # Calculation of final scores for initial candidates, self.initial_population
        final_scores = {}
        for idx, candidate in enumerate(population):
            candidate_key = tuple(candidate)
            children = self.parent_child_mapping.get(candidate_key, [])
            children_scores = [self.fitness_function(child) for child in children]
            max_child_score = max(children_scores) if children_scores else 0

            # Final score = max(initial score, virtual mutation score, best child score)
            final_score = max(
                self.fitness_function(candidate),
                virtual_scores[idx],
                max_child_score
            )
            # Retrieve the ID
            candidate_id = self.candidate_id_map[candidate_key]
            final_scores[candidate_id] = final_score

        # Return the list of initial candidates ranked by their final score
        # and the dictionary of final scores (key = ID, value = score).
        # To reconstruct the vector, use self.id_to_vector_map[ID]. self.vector_to_user.get(tuple(
        # print("Tous les users: ", self.id_to_vector_map)    
        sorted_ids = sorted(final_scores, key=final_scores.get, reverse=True)

        ranked_candidates = []
        for cid in sorted_ids:
            # user = await get_user_by_id(cid, users_collection)
            raw_user = await get_user_by_id(cid, users_collection)
            user = serialize_mongo_document(raw_user)
            ranked_candidates.append((cid, user, final_scores[cid]))

        return ranked_candidates, final_scores

def plot_all_candidates_evolution_with_id(history):
    """
    Plots the evolution of each candidate (identified by candidate_id) across all generations.
    Points (and lines) are colored based on candidate_id.
    
    Hovering over a point displays:
      - Candidate ID
      - Generation
      - Raw score, bonus, final score
      - Operation and description (specific to the action performed)
    """
    final_sel = history.get('final_selection', [])
    if not final_sel:
        print("No final selection recorded.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(final_sel)
    
    # Filter to keep only numeric generations
    df = df[df['generation'].apply(lambda x: isinstance(x, int))]
    
    # Ensure final_score is numeric
    df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
    
    # Sort by generation
    df = df.sort_values(by='generation')
    
    # Update descriptions and operations based on actions
    for idx, row in df.iterrows():
        if row['bonus'] == 0:
            df.at[idx, 'description'] = "No operation performed."
            df.at[idx, 'operation'] = "No operation"
        elif row['bonus'] == 0.05:
            df.at[idx, 'description'] = "Only mutation performed."
            df.at[idx, 'operation'] = "Mutation"
        elif row['bonus'] == 0.1:
            df.at[idx, 'description'] = "Only crossover performed."
            df.at[idx, 'operation'] = "Crossover"
        elif row['bonus'] == 0.15:
            df.at[idx, 'description'] = "Both mutation and crossover performed."
            df.at[idx, 'operation'] = "Mutation + Crossover"
    
    # Build a line plot where each 'candidate_id' will have a unique color
    fig = px.line(
        df,
        x='generation',
        y='final_score',
        color='candidate_id',
        markers=True,
        labels={
            'generation': 'Generation',
            'final_score': 'Final Score',
            'candidate_id': 'Candidate ID'
        },
        title="Evolution of Final Scores for Each Candidate (via Unique ID)"
    )
    
    # Customize hover to display additional info
    fig.update_traces(
        mode="markers+lines",
        hovertemplate=(
            "Candidate ID: %{customdata[0]}<br>"
            "Generation: %{x}<br>"
            "Raw Score: %{customdata[1]:.3f}<br>"
            "Bonus: %{customdata[2]:.3f}<br>"
            "Final Score: %{y:.3f}<br>"
            "Operation: %{customdata[3]}<br>"
            "Description: %{customdata[4]}<extra></extra>"
        ),
        # Pass raw data, bonus, operation, etc. to hover
        customdata=df[['candidate_id', 'raw_score', 'bonus', 'operation', 'description']].values
    )
    
    # Highlight the best score in the last generation
    last_gen = df['generation'].max()
    last_gen_data = df[df['generation'] == last_gen]
    if not last_gen_data.empty:
        best_idx = last_gen_data['final_score'].idxmax()
        best_score = last_gen_data.loc[best_idx, 'final_score']
        best_cand_id = last_gen_data.loc[best_idx, 'candidate_id']
        
        fig.add_trace(go.Scatter(
            x=[last_gen],
            y=[best_score],
            mode='markers+text',
            marker=dict(color='gold', size=15, symbol='star'),
            text=[f"Best ID: {best_cand_id}"],
            textposition="top center",
            name="Best Final Score"
        ))
    
    # Force display of increments of 1 on the generation axis
    fig.update_layout(xaxis=dict(dtick=1))
    fig.show()