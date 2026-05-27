#!/usr/bin/env python3
"""
Task Generation Module

This module defines Task classes and utilities for generating prediction tasks
on RDB tables. Currently supports single-table column prediction tasks.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Set, Any
from enum import Enum
from dataclasses import dataclass

from .dataset_meta import (
    DBBTaskType,
    DBBTaskEvalMetric,
    DBBTaskMeta,
    DBBColumnSchema,
    DBBColumnDType,
    DBBTableDataFormat,
)
from .table_generation import (
    convert_to_dbb_column_dtype,
    DataType,
    Table,
    TaskGenerationSchema,
    Relationship,
)
from .task_generation_utils import (
    TaskType,
    AggregationFunction,
    PredicateFunction,
    PredicateOperator,
    TargetNodeSet,
    DirectAttributeTarget,
    RelationalAggregationTarget,
    SchemaEdge,
    SchemaEdgeDirection,
    SchemaGraph,
    AggregationFunctionList,
    PredicateFunctionList,
)


class Task:
    """
    A task defines a prediction problem on RDB data.
    Each task is associated with key tables and a specific prediction target.
    """

    def __init__(
        self,
        task_name: str,
        task_type: TaskType,
        key_tables: Union[str, List[str]],
        target_table: str,
        real_name_for_target_table: str,
        target_column: str,
        real_name_for_target_column: str,
        evaluation_metric: DBBTaskEvalMetric,
        dbb_task_type: DBBTaskType,
        num_classes: Optional[int] = None,
        description: Optional[str] = None,
        time_column_name: Optional[str] = None,
        # Below are optional fields for complex tasks
        target_node_set: Optional[TargetNodeSet] = None,
        target_computation: Union[
            DirectAttributeTarget, RelationalAggregationTarget
        ] = None,
        schema_graph: Optional[SchemaGraph] = None,
    ):
        """
        Initialize a Task.

        Parameters
        ----------
        task_name : str
            Unique name for the task
        task_type : TaskType
            Type of the task (currently only SINGLE_TABLE_PREDICTION)
        key_tables : Union[str, List[str]]
            Key table(s) for this task. For single table prediction, this is a single table name.
            For future multi-table tasks, this can be a list of table names.
        target_table : str
            Name of the table containing the target column
        real_name_for_target_table : str
            Name of the table containing the actual primary key column
        target_column : str
            Name of the column to predict. Note that this is always the column name in the target table.
        real_name_for_target_column : str
            Name of the column to predict in the generated task table
        evaluation_metric : DBBTaskEvalMetric
            Evaluation metric for the task
        dbb_task_type : DBBTaskType
            4DBInfer task type (classification/regression)
        num_classes : Optional[int]
            Number of classes for classification tasks
        description : Optional[str]
            Optional description of the task
        time_column_name : Optional[str]
            Name of the time column in the key table
        """
        self.task_name = task_name
        self.task_type = task_type

        # Handle key tables - normalize to list format
        if isinstance(key_tables, str):
            self.key_tables = [key_tables]
        else:
            self.key_tables = key_tables

        self.target_table = target_table
        self.real_name_for_target_table = real_name_for_target_table
        self.target_column = target_column
        self.real_name_for_target_column = real_name_for_target_column
        self.evaluation_metric = evaluation_metric
        self.dbb_task_type = dbb_task_type
        self.num_classes = num_classes
        self.description = description
        self.time_column_name = time_column_name
        self.target_node_set = target_node_set
        self.target_computation = target_computation
        self.schema_graph = schema_graph

        # Validate that target table is in key tables for single table prediction
        if self.task_type == TaskType.SINGLE_TABLE_PREDICTION:
            if len(self.key_tables) != 1:
                raise ValueError(
                    "Single table prediction requires exactly one key table"
                )
            if self.target_table != self.key_tables[0]:
                raise ValueError(
                    "For single table prediction, target table must be the same as key table"
                )
        elif self.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION:
            if self.target_computation is None:
                raise ValueError(
                    "Direct attribute prediction requires a target computation"
                )
        elif self.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION:
            if self.target_computation is None:
                raise ValueError(
                    "Relational aggregation prediction requires a target computation"
                )

        # Task metadata and data will be generated by TaskDataGenerator
        self.task_metadata = None  # Stores column specifications, features vs labels
        self.unified_dataframe = None  # Combined features and labels dataframe
        self.task_data = None  # Train/validation/test splits

    def __repr__(self):
        return (
            f"Task(name={self.task_name}, type={self.task_type}, "
            f"key_tables={self.key_tables}, target={self.target_table}.{self.target_column}, "
            f"metric={self.evaluation_metric}), target_node_set={self.target_node_set}, target_computation={self.target_computation})"
        )

    def is_classification_task(self) -> bool:
        """Check if this is a classification task"""
        return self.dbb_task_type == DBBTaskType.classification

    def is_regression_task(self) -> bool:
        """Check if this is a regression task"""
        return self.dbb_task_type == DBBTaskType.regression

    def get_primary_key_table(self) -> str:
        """Get the primary key table for this task (first key table)"""
        return self.key_tables[0]

    def set_task_column_metadata(self, metadata: Dict):
        """
        Set task metadata generated by TaskDataGenerator.

        Parameters
        ----------
        metadata : Dict
            Task metadata containing column specifications and feature/label info
        """
        self.task_metadata = metadata

    def set_unified_dataframe(self, dataframe: pd.DataFrame):
        """
        Set unified dataframe containing all required features and labels.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Unified dataframe with features and labels
        """
        self.unified_dataframe = dataframe

    def set_task_data(self, task_data: Dict):
        """
        Set task data generated by TaskDataGenerator.

        Parameters
        ----------
        task_data : Dict
            Task data dictionary containing splits and features
        """
        self.task_data = task_data

    def to_dbb_task_meta(self, source_file: str) -> DBBTaskMeta:
        """
        Convert this task to 4DBInfer task metadata format.

        Parameters
        ----------
        source_file : str
            Path to the task data file

        Returns
        -------
        DBBTaskMeta
            Task metadata in 4DBInfer format
        """
        # Create column schemas for the task data
        column_schemas = []
        for k, v in self.task_metadata["column_metadata"].items():
            column_schema = DBBColumnSchema(
                name=k,
                dtype=convert_to_dbb_column_dtype(v["data_type"]),
                **({"in_size": 1} if v["data_type"] == DataType.FLOAT else {}),
                **(
                    {"num_categories": v["num_categories"]}
                    if v["data_type"] == DataType.CATEGORICAL
                    else {}
                ),
                **(
                    {
                        # * Here we assume the parent table's primary key name is the same as the foreign key name
                        # * This is not always true for real table, but it is true for the generated synthetic tables
                        "link_to": f"{v['parent_table']}.{v['name']}"
                    }
                    if v["data_type"] == DataType.FOREIGN_KEY
                    else {}
                ),
            )
            column_schemas.append(column_schema)

        if (
            self.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION
            or self.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION
        ):
            column_schemas.append(
                DBBColumnSchema(
                    name=self.real_name_for_target_column,
                    dtype=DBBColumnDType.category_t,
                )
            )

        # Create task metadata
        task_meta = DBBTaskMeta(
            name=self.task_name,
            source=source_file,
            format=DBBTableDataFormat.PARQUET,
            columns=column_schemas,
            evaluation_metric=self.evaluation_metric,
            target_column=self.real_name_for_target_column,
            target_table=self.real_name_for_target_table,
            task_type=self.dbb_task_type,
            time_column=self.time_column_name,
        )

        if self.is_classification_task() and self.num_classes:
            task_meta.num_classes = self.num_classes

        return task_meta


class TaskDataGenerator:
    """
    Generates task data from key tables for specific task types.
    Handles metadata generation, feature combination, and data splitting separately.
    """

    def __init__(self, rdb, random_seed: int = 42):
        """
        Initialize TaskDataGenerator.

        Parameters
        ----------
        rdb : RDB
            The RDB object containing the tables
        random_seed : int
            Random seed for reproducible data generation
        """
        self.rdb = rdb
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _compute_column_metadata(
        self,
        task: Task,
        key_table: Table,
        all_columns: List[str],
        feature_columns: List[str],
        label_column: str,
    ) -> Dict:
        """
        Compute column metadata for a task.
        """
        column_metadata = {}
        for i, col_name in enumerate(all_columns):
            data_type_config = key_table.data_type_configs[i]
            col_meta = {
                "name": col_name,
                "data_type": data_type_config.data_type,
                "is_feature": col_name in feature_columns,
                "is_label": col_name == label_column,
                "column_index": i,
            }
            if data_type_config.data_type == DataType.CATEGORICAL:
                col_meta["num_categories"] = data_type_config.config.get(
                    "num_categories", 2
                )
            elif data_type_config.data_type == DataType.FOREIGN_KEY:
                col_meta["parent_table"] = data_type_config.config.get("parent_table")
            column_metadata[col_name] = col_meta
        return column_metadata

    def generate_task_column_metadata(
        self,
        task: Task,
        key_tables: Dict[str, Table],
    ) -> Dict:
        """
        Generate task column metadata specifying columns, features, and labels.

        Parameters
        ----------
        task : Task
            Task object containing task specifications
        key_tables : Dict[str, Table]
            Dictionary mapping table names to Table objects

        Returns
        -------
        Dict
            Task column metadata containing column specifications
        """
        if task.task_type == TaskType.SINGLE_TABLE_PREDICTION:
            # Get the single key table
            key_table_name = task.key_tables[0]
            if key_table_name not in key_tables:
                raise ValueError(f"Key table '{key_table_name}' not found")

            key_table = key_tables[key_table_name]
            all_columns = key_table.column_names.copy()
            feature_columns = [col for col in all_columns if col != task.target_column]
            label_column = task.target_column

            column_metadata = self._compute_column_metadata(
                task, key_table, all_columns, feature_columns, label_column
            )

            task_column_metadata = {
                "key_table": key_table_name,
                "all_columns": all_columns,
                "feature_columns": feature_columns,
                "label_column": label_column,
                "column_metadata": column_metadata,
            }

            return task_column_metadata
        elif (
            task.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION
            or task.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION
        ):
            # For complex tasks, joined columns are not known until
            # _join_related_features runs.  Compute metadata from the focal
            # table's own columns here; joined columns are patched in by
            # generate_task_data after combine_features_and_labels.
            key_table_name = task.key_tables[0]
            if key_table_name not in key_tables:
                raise ValueError(f"Key table '{key_table_name}' not found")

            key_table = key_tables[key_table_name]
            all_columns = key_table.column_names.copy()
            feature_columns = [col for col in all_columns if col != task.target_column]
            label_column = task.target_column

            column_metadata = self._compute_column_metadata(
                task, key_table, all_columns, feature_columns, label_column
            )

            task_column_metadata = {
                "key_table": key_table_name,
                "all_columns": all_columns,
                "feature_columns": feature_columns,
                "label_column": label_column,
                "column_metadata": column_metadata,
                # Filled by generate_task_data after joins.
                "_joined_cols_pending": True,
            }

            return task_column_metadata

    def _get_joined_feature_columns(
        self,
        focal_table_name: str,
        schema_graph: "SchemaGraph",
    ) -> List[Tuple[str, "DataType"]]:
        """Return ``(column_name, DataType)`` for columns ``_join_related_features`` adds."""
        if schema_graph is None or len(schema_graph.nodes) <= 1:
            return []

        topo_order = schema_graph.get_topological_order()
        remaining = [t for t in topo_order if t != focal_table_name]
        if not remaining:
            return []

        joined_cols: List[Tuple[str, DataType]] = []
        joined_tables = {focal_table_name}

        for table_name in remaining:
            # Find any already-joined table that has an edge with table_name
            # (same BFS logic as _join_related_features).
            linked_name = None
            for neighbor in schema_graph.adjacency.get(table_name, []):
                if neighbor in joined_tables:
                    linked_name = neighbor
                    break
            if linked_name is None:
                for t, neighbors in schema_graph.adjacency.items():
                    if t in joined_tables and table_name in neighbors:
                        linked_name = t
                        break

            other_table = self.rdb.tables[table_name]
            feat_cols = other_table.get_feature_columns()

            if linked_name is not None:
                edge = schema_graph.get_edge(linked_name, table_name)
                if edge is None:
                    edge = schema_graph.get_edge(table_name, linked_name)

                if edge is not None:
                    if edge.direction == SchemaEdgeDirection.FK_TO_PK:
                        col_name_to_idx = {
                            name: i for i, name in enumerate(other_table.column_names)
                        }
                        for c in feat_cols:
                            idx = col_name_to_idx.get(c)
                            col_dtype = (
                                other_table.data_type_configs[idx].data_type
                                if idx is not None
                                else DataType.FLOAT
                            )
                            joined_cols.append((f"{table_name}_{c}", col_dtype))

                    elif edge.direction == SchemaEdgeDirection.PK_TO_FK:
                        for c in feat_cols:
                            joined_cols.append(
                                (f"{table_name}_{c}_mean", DataType.FLOAT)
                            )
                            joined_cols.append(
                                (f"{table_name}_{c}_std", DataType.FLOAT)
                            )

                    joined_tables.add(table_name)
                    continue

            # ── multi-hop bridge (mirrors _join_related_features) ──────────
            if not feat_cols:
                continue

            for mid_name, mid_table in self.rdb.tables.items():
                if mid_name == table_name or mid_name == focal_table_name:
                    continue

                # T FK → M PK?
                t_fk_col, _, _ = self._find_fk_to(other_table, mid_name)
                if t_fk_col is None:
                    continue

                # M FK → J (any already-joined table)?
                bridge_found = False
                for j_name in joined_tables:
                    m_fk, _, _ = self._find_fk_to(mid_table, j_name)
                    if m_fk is not None:
                        bridge_found = True
                        break
                if not bridge_found:
                    continue

                # Bridge join always aggregates (mean/std).
                for c in feat_cols:
                    joined_cols.append(
                        (f"{table_name}_{c}_mean", DataType.FLOAT)
                    )
                    joined_cols.append(
                        (f"{table_name}_{c}_std", DataType.FLOAT)
                    )
                break
            joined_tables.add(table_name)

        return joined_cols

    def _join_related_features(
        self,
        combined_df: pd.DataFrame,
        focal_table_name: str,
        schema_graph: "SchemaGraph",
    ) -> pd.DataFrame:
        """Join feature columns from related tables via FK relationships.

        Traverses the schema graph from the focal table outward, joining
        feature columns (float + categorical) from each non-focal table.
        Column names are prefixed with ``{table_name}_`` to avoid collisions.
        """
        if schema_graph is None or len(schema_graph.nodes) <= 1:
            return combined_df

        topo_order = schema_graph.get_topological_order()
        remaining = [t for t in topo_order if t != focal_table_name]
        if not remaining:
            return combined_df

        orig_index = combined_df.index
        joined_tables = {focal_table_name}

        for table_name in remaining:
            # Find any already-joined table that has an edge with table_name.
            linked_name = None
            for neighbor in schema_graph.adjacency.get(table_name, []):
                if neighbor in joined_tables:
                    linked_name = neighbor
                    break
            if linked_name is None:
                # Also check reverse adjacency
                for t, neighbors in schema_graph.adjacency.items():
                    if t in joined_tables and table_name in neighbors:
                        linked_name = t
                        break
            other_table = self.rdb.tables[table_name]
            if other_table.dataframe is None:
                other_table.generate_dataframe()

            feat_cols = other_table.get_feature_columns()
            # Also pull FK columns so they are available for subsequent joins.
            fk_cols = [
                other_table.column_names[i]
                for i, dtc in enumerate(other_table.data_type_configs)
                if dtc.data_type == DataType.FOREIGN_KEY
            ]
            extra_cols = [c for c in fk_cols if c not in feat_cols]

            # ── direct edge join ──────────────────────────────────────────
            if linked_name is not None:
                edge = schema_graph.get_edge(linked_name, table_name)
                if edge is None:
                    edge = schema_graph.get_edge(table_name, linked_name)
                if edge is not None and (feat_cols or extra_cols):
                    t_linked = self.rdb.tables[linked_name]
                    t_other = other_table

                    # Case 1: linked_name FK → table_name PK (direct merge)
                    fk_col, pk_table, _ = self._find_fk_to(t_linked, table_name)
                    if fk_col is not None and pk_table is not None:
                        fk_col_name = t_linked.column_names[fk_col]
                        pk_col_name = t_other.column_names[0]
                        if fk_col_name in combined_df.columns:
                            keep_cols = [pk_col_name] + feat_cols + extra_cols
                            other_feat_df = t_other.dataframe[keep_cols].copy()
                            rename_map = {c: f"{table_name}_{c}" for c in feat_cols}
                            other_feat_df = other_feat_df.rename(columns=rename_map)
                            if other_feat_df[pk_col_name].duplicated().any():
                                other_feat_df = other_feat_df.drop_duplicates(
                                    subset=[pk_col_name], keep="first"
                                )
                            combined_df = combined_df.merge(
                                other_feat_df,
                                left_on=fk_col_name,
                                right_on=pk_col_name,
                                how="left",
                            )
                            combined_df.index = orig_index
                            joined_tables.add(table_name)
                            continue

                    # Case 2: table_name FK → linked_name PK (aggregate)
                    fk_col2, pk_table2, _ = self._find_fk_to(t_other, linked_name)
                    if fk_col2 is not None and pk_table2 is not None:
                        fk_col_name2 = t_other.column_names[fk_col2]
                        pk_col_name2 = t_linked.column_names[0]
                        if pk_col_name2 in combined_df.columns:
                            child_df = t_other.dataframe[[fk_col_name2] + feat_cols].copy()
                            agg_funcs = {c: ["mean", "std"] for c in feat_cols}
                            agg_df = child_df.groupby(fk_col_name2, as_index=False).agg(agg_funcs)
                            agg_df.columns = [
                                fk_col_name2
                                if col[0] == fk_col_name2
                                else f"{table_name}_{col[0]}_{col[1]}"
                                for col in agg_df.columns
                            ]
                            combined_df = combined_df.merge(
                                agg_df,
                                left_on=pk_col_name2,
                                right_on=fk_col_name2,
                                how="left",
                            )
                            combined_df.index = orig_index
                            joined_tables.add(table_name)
                            continue

            # ── multi-hop bridge join ─────────────────────────────────────
            # When linked_name is None (no direct edge to a joined table),
            # try a 2-hop path: table T has FK→M, and M has FK→J (J already
            # joined).  Needed because Case 2 aggregation drops M's PK from
            # combined_df, blocking direct FK joins from grandchild tables.
            if not feat_cols:
                continue

            t_other = other_table
            for mid_name, mid_table in self.rdb.tables.items():
                if mid_name == table_name or mid_name == focal_table_name:
                    continue
                if mid_table.dataframe is None:
                    mid_table.generate_dataframe()

                # T FK → M PK?
                t_fk_col, _, _ = self._find_fk_to(t_other, mid_name)
                if t_fk_col is None:
                    continue

                # M FK → J (any already-joined table)?
                bridge_linked = None
                bridge_fk_col = None
                for j_name in joined_tables:
                    m_fk, _, _ = self._find_fk_to(mid_table, j_name)
                    if m_fk is not None:
                        bridge_linked = j_name
                        bridge_fk_col = m_fk
                        break
                if bridge_linked is None:
                    continue

                t_fk_col_name = t_other.column_names[t_fk_col]
                bridge_fk_col_name = mid_table.column_names[bridge_fk_col]
                bridge_pk_col_name = mid_table.column_names[0]
                t_linked2 = self.rdb.tables[bridge_linked]
                pk_col_name2 = t_linked2.column_names[0]
                if pk_col_name2 not in combined_df.columns:
                    continue

                # 1. Merge T with bridge M to propagate bridge FK.
                t_df = t_other.dataframe[[t_fk_col_name] + feat_cols].copy()
                bridge_df = mid_table.dataframe[
                    [bridge_pk_col_name, bridge_fk_col_name]
                ].copy()
                bridge_df = bridge_df.rename(
                    columns={bridge_pk_col_name: "_bridge_pk"}
                )
                merged = t_df.merge(
                    bridge_df,
                    left_on=t_fk_col_name,
                    right_on="_bridge_pk",
                    how="left",
                )

                # 2. Aggregate T features by bridge FK.
                agg_funcs = {c: ["mean", "std"] for c in feat_cols}
                agg_df = merged.groupby(
                    bridge_fk_col_name, as_index=False
                ).agg(agg_funcs)
                agg_df.columns = [
                    bridge_fk_col_name
                    if col[0] == bridge_fk_col_name
                    else f"{table_name}_{col[0]}_{col[1]}"
                    for col in agg_df.columns
                ]

                # 3. Join via J.PK = bridge FK.
                combined_df = combined_df.merge(
                    agg_df,
                    left_on=pk_col_name2,
                    right_on=bridge_fk_col_name,
                    how="left",
                )
                combined_df.index = orig_index
                joined_tables.add(table_name)
                break
            else:
                # No bridge found either — add to joined set anyway so
                # subsequent tables can still try to link through it.
                joined_tables.add(table_name)

        return combined_df

    @staticmethod
    def _find_fk_to(
        table: "Table", target_table_name: str
    ) -> Tuple[int | None, str | None, int | None]:
        """Return ``(fk_col_idx, pk_table_name, pk_col_idx)`` if *table* has
        an FK pointing to *target_table_name*, else ``(None, None, None)``."""
        for i, dtc in enumerate(table.data_type_configs):
            if dtc.data_type == DataType.FOREIGN_KEY:
                parent = dtc.config.get("parent_table")
                if parent == target_table_name:
                    return (i, target_table_name, 0)
        return (None, None, None)

    def _walk_and_merge(
        self,
        df: pd.DataFrame,
        start_table: str,
        path: List[str],
        schema_graph: "SchemaGraph",
    ) -> pd.DataFrame:
        """Walk FK edges along *path*, merging only PK and FK columns.

        Returns a DataFrame that maps the start table's PK to the terminal
        table's PK (plus intermediate FK columns).  No feature columns are
        carried — the caller merges those separately.
        """
        current_df = df
        current_table = start_table
        active_pk = self.rdb.tables[start_table].column_names[0]

        for i, next_table in enumerate(path):
            is_terminal = (i == len(path) - 1)
            edge = schema_graph.get_edge(current_table, next_table)
            if edge is None:
                edge = schema_graph.get_edge(next_table, current_table)
            if edge is None:
                raise ValueError(
                    f"No edge between {current_table} and {next_table}"
                )

            next_tbl = self.rdb.tables[next_table]
            if next_tbl.dataframe is None:
                next_tbl.generate_dataframe()

            pk_tag = f"_pk_{next_table}"

            # FK columns of *next_table* needed when it becomes the current
            # table in a subsequent PK_TO_FK step.
            next_fk_indices: list[int] = []
            if not is_terminal:
                for j, dtc in enumerate(next_tbl.data_type_configs):
                    if dtc.data_type == DataType.FOREIGN_KEY and j != 0:
                        next_fk_indices.append(j)

            if edge.direction == SchemaEdgeDirection.PK_TO_FK:
                # current.FK  =  next.PK
                cur_fk_idx = edge.from_column
                cur_fk_name = self.rdb.tables[current_table].column_names[
                    cur_fk_idx
                ]
                col_idxs = [0] + next_fk_indices
                next_df = next_tbl.dataframe.iloc[:, col_idxs].copy()
                next_names = [pk_tag] + [
                    next_tbl.column_names[j] for j in next_fk_indices
                ]
                next_df.columns = next_names
                current_df = current_df.merge(
                    next_df, left_on=cur_fk_name, right_on=pk_tag,
                    how="left",
                )
            else:  # FK_TO_PK
                # current.PK  =  next.FK
                next_fk_idx = edge.to_column
                next_fk_tag = f"_fk_{next_table}"
                col_idxs = [0, next_fk_idx] + [
                    j for j in next_fk_indices if j != next_fk_idx
                ]
                next_df = next_tbl.dataframe.iloc[:, col_idxs].copy()
                next_names = [pk_tag, next_fk_tag] + [
                    next_tbl.column_names[j]
                    for j in next_fk_indices if j != next_fk_idx
                ]
                next_df.columns = next_names
                current_df = current_df.merge(
                    next_df, left_on=active_pk, right_on=next_fk_tag,
                    how="left",
                )

            current_table = next_table
            active_pk = pk_tag

        return current_df

    def _lookup_terminal_column(
        self,
        merged: pd.DataFrame,
        focal_pk: str,
        target_table_name: str,
        column_name: str,
    ) -> pd.Series:
        """Merge terminal-table column into *merged* via the terminal PK."""
        target_tbl = self.rdb.tables[target_table_name]
        if target_tbl.dataframe is None:
            target_tbl.generate_dataframe()
        target_pk_tag = f"_pk_{target_table_name}"
        target_pk = target_tbl.column_names[0]

        target_df = target_tbl.dataframe[[target_pk, column_name]].copy()
        target_df = target_df.rename(columns={target_pk: target_pk_tag})
        return merged.merge(target_df, on=target_pk_tag, how="left")

    def _compute_direct_attribute_labels_bulk(
        self,
        task: Task,
        key_table_name: str,
        key_table: Table,
    ) -> pd.DataFrame:
        """Bulk DirectAttribute label computation — no per-row InstanceGraph."""
        target_comp = task.target_computation
        target_table_name = target_comp.table_name

        focal_df = key_table.dataframe.copy()
        focal_pk = key_table.column_names[0]

        result = self._join_related_features(
            focal_df, key_table_name, task.schema_graph,
        )

        if target_table_name == key_table_name:
            labels = key_table.dataframe[target_comp.column_name].values
        else:
            path = task.schema_graph.find_path(key_table_name, target_table_name)
            if path is None:
                return None
            init_cols = self._initial_walk_cols(
                key_table_name, path[0], task.schema_graph, focal_pk,
            )
            merged = self._walk_and_merge(
                focal_df[init_cols], key_table_name, path, task.schema_graph,
            )
            merged = self._lookup_terminal_column(
                merged, focal_pk, target_table_name, target_comp.column_name,
            )
            labels = (
                merged.groupby(focal_pk)[target_comp.column_name]
                .first()
                .reindex(focal_df[focal_pk])
                .values
            )

        result[task.real_name_for_target_column] = labels
        return result

    def _compute_aggregation_labels_bulk(
        self,
        task: Task,
        key_table_name: str,
        key_table: Table,
        max_retries: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Bulk RelationalAggregation label computation via merge + groupby."""
        target_comp = task.target_computation
        target_node_set = target_comp.target_node_set
        target_table_name = target_node_set.table_name

        focal_df = key_table.dataframe.copy()
        focal_pk = key_table.column_names[0]

        result = self._join_related_features(
            focal_df, key_table_name, task.schema_graph,
        )

        if target_table_name == key_table_name:
            lookup_df = focal_df[[focal_pk, target_comp.aggregation_column]].copy()
        else:
            path = task.schema_graph.find_path(key_table_name, target_table_name)
            if path is None:
                return None
            init_cols = self._initial_walk_cols(
                key_table_name, path[0], task.schema_graph, focal_pk,
            )
            merged = self._walk_and_merge(
                focal_df[init_cols], key_table_name, path, task.schema_graph,
            )

        for attempt in range(max_retries + 1):
            agg_col = target_comp.aggregation_column
            agg_func = target_comp.aggregation_func

            if target_table_name != key_table_name:
                lookup_df = self._lookup_terminal_column(
                    merged, focal_pk, target_table_name, agg_col,
                )

            if agg_col not in lookup_df.columns:
                return None

            # Map AggregationFunction to a pandas groupby operation.
            _agg_name_map = {
                "COUNT": "count", "SUM": "sum", "AVG": "mean",
                "MAX": "max", "MIN": "min", "STD": "std",
            }
            _agg_name = _agg_name_map.get(agg_func.name, "mean")
            grouped = lookup_df.groupby(focal_pk)[agg_col].agg(_agg_name)
            valid = grouped.dropna()
            valid = valid[~np.isinf(valid.values)]

            if len(valid) < 2 or len(set(valid.values)) <= 1:
                if attempt == max_retries:
                    return None
                target_table = self.rdb.tables[target_table_name]
                float_cols = target_table.get_feature_columns(only_float=True)
                if not float_cols:
                    return None
                target_comp.aggregation_column = random.choice(float_cols)
                target_comp.aggregation_func = random.choice(
                    AggregationFunctionList,
                )
                template = random.choice(PredicateFunctionList)
                target_comp.predicate_func = PredicateFunction(
                    template.operator, template.threshold,
                )
                continue

            threshold = float(np.median(valid.values))
            target_comp.predicate_func = PredicateFunction(
                target_comp.predicate_func.operator, threshold,
            )
            labels = valid.apply(
                lambda v: int(target_comp.predicate_func.apply(v))
            )

            if len(labels) > 0 and len(set(labels.values)) > 1:
                break
        else:
            return None

        result[task.real_name_for_target_column] = (
            labels.reindex(focal_df[focal_pk]).fillna(0).astype(int).values
        )
        return result

    def _initial_walk_cols(
        self,
        start_table: str,
        first_next: str,
        schema_graph: "SchemaGraph",
        pk_col: str,
    ) -> list[str]:
        """Columns needed in the initial DataFrame for ``_walk_and_merge``."""
        edge = schema_graph.get_edge(start_table, first_next)
        if edge is None:
            edge = schema_graph.get_edge(first_next, start_table)
        if edge is not None and edge.direction == SchemaEdgeDirection.PK_TO_FK:
            fk_col = self.rdb.tables[start_table].column_names[
                edge.from_column
            ]
            return [pk_col, fk_col]
        return [pk_col]

    def combine_features_and_labels(
        self,
        task: Task,
        key_tables: Dict[str, Table],
    ) -> pd.DataFrame:
        """Combine required features and labels into a unified dataframe."""
        if task.task_metadata is None:
            raise ValueError(
                "Task metadata not generated. Call generate_task_metadata first."
            )

        key_table_name = task.task_metadata["key_table"]
        key_table = key_tables[key_table_name]

        if key_table.dataframe is None:
            key_table.generate_dataframe()

        if task.task_type == TaskType.SINGLE_TABLE_PREDICTION:
            required_columns = task.task_metadata["feature_columns"] + [
                task.task_metadata["label_column"]
            ]
            unified_df = key_table.dataframe[required_columns].copy()

        elif task.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION:
            unified_df = self._compute_direct_attribute_labels_bulk(
                task, key_table_name, key_table,
            )

        elif task.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION:
            unified_df = self._compute_aggregation_labels_bulk(
                task, key_table_name, key_table,
            )

        return unified_df

    def split_task_data(
        self,
        task: Task,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> Dict:
        """
        Split unified dataframe into train/validation/test sets.

        Parameters
        ----------
        task : Task
            Task object with unified dataframe
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation

        Returns
        -------
        Dict
            Task data dictionary containing splits
        """
        if task.unified_dataframe is None:
            raise ValueError(
                "Unified dataframe not created. Call combine_features_and_labels first."
            )

        if task.task_metadata is None:
            raise ValueError(
                "Task metadata not generated. Call generate_task_metadata first."
            )

        num_rows = len(task.unified_dataframe)
        num_train = int(num_rows * train_ratio)
        num_valid = int(num_rows * valid_ratio)

        # Create random indices for train/valid/test split
        indices = torch.randperm(num_rows)
        train_indices = indices[:num_train]
        valid_indices = indices[num_train : num_train + num_valid]
        test_indices = indices[num_train + num_valid :]

        # Extract target column (labels)
        real_target_column = task.real_name_for_target_column
        all_labels = task.unified_dataframe[real_target_column]

        # Create input features (all columns except target)
        feature_columns = task.task_metadata["feature_columns"]
        all_features = task.unified_dataframe[feature_columns]

        # Split into train/valid/test
        train_features = all_features.iloc[train_indices]
        train_labels = all_labels.iloc[train_indices]
        valid_features = all_features.iloc[valid_indices]
        valid_labels = all_labels.iloc[valid_indices]
        test_features = all_features.iloc[test_indices]
        test_labels = all_labels.iloc[test_indices]

        # Full dataframe splits
        train_df = task.unified_dataframe.iloc[train_indices]
        validation_df = task.unified_dataframe.iloc[valid_indices]
        test_df = task.unified_dataframe.iloc[test_indices]

        # Store task data
        task_data = {
            "train_features": train_features,
            "train_labels": train_labels,
            "validation_features": valid_features,
            "validation_labels": valid_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "train_indices": train_indices,
            "validation_indices": valid_indices,
            "test_indices": test_indices,
            "train_df": train_df,
            "test_df": test_df,
            "validation_df": validation_df,
            "feature_columns": feature_columns,
            "target_column": real_target_column,
        }

        return task_data

    def generate_task_data(
        self,
        task: Task,
        key_tables: Dict[str, Table],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> Dict:
        """
        Generate complete task data: metadata -> unified dataframe -> splits.

        Parameters
        ----------
        task : Task
            Task object containing task specifications
        key_tables : Dict[str, Table]
            Dictionary mapping table names to Table objects
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation

        Returns
        -------
        Dict
            Task data dictionary containing splits and features
        """
        # Step 1: Generate task metadata
        task_column_metadata = self.generate_task_column_metadata(task, key_tables)
        task.set_task_column_metadata(task_column_metadata)

        # Step 2: Combine features and labels into unified dataframe
        unified_df = self.combine_features_and_labels(task, key_tables)
        if unified_df is None:
            return None
        task.set_unified_dataframe(unified_df)

        # If joined columns were added, patch the metadata.
        if task_column_metadata.get("_joined_cols_pending"):
            del task_column_metadata["_joined_cols_pending"]
            all_cols = list(unified_df.columns)
            focal_cols = task_column_metadata["all_columns"]
            joined_cols = [c for c in all_cols if c not in focal_cols and c != task.real_name_for_target_column]
            for col_name in joined_cols:
                task_column_metadata["all_columns"].append(col_name)
                task_column_metadata["feature_columns"].append(col_name)
                task_column_metadata["column_metadata"][col_name] = {
                    "name": col_name,
                    "data_type": DataType.FLOAT,
                    "is_feature": True,
                    "is_label": False,
                    "column_index": -1,
                }
            task.set_task_column_metadata(task_column_metadata)

        # Step 3: Split the unified dataframe
        task_data = self.split_task_data(task, train_ratio, valid_ratio)

        return task_data

    def save_task_data(self, task_data: Dict, file_path: str):
        """
        Save task data to files in 4DBInfer format.

        Parameters
        ----------
        task_data : Dict
            Task data dictionary
        file_path : str
            Directory path to save task data
        """
        if task_data is None:
            raise ValueError("Task data is None")

        os.makedirs(file_path, exist_ok=True)

        # Save train/valid/test data
        task_data["train_df"].to_parquet(
            os.path.join(file_path, "train.parquet"), index=False
        )
        task_data["validation_df"].to_parquet(
            os.path.join(file_path, "validation.parquet"), index=False
        )
        task_data["test_df"].to_parquet(
            os.path.join(file_path, "test.parquet"), index=False
        )


class TaskGenerator:
    """
    Main interface for generating tasks and task data for RDB tables.
    Uses TaskDataGenerator internally for data generation.
    """

    def __init__(
        self,
        rdb,
        random_seed: int = 42,
        entity_task_ratio: float = 0.75,
        relbench_mode: bool = False,
    ):
        """
        Initialize TaskGenerator.

        Parameters
        ----------
        rdb : RDB
            RDB object
        random_seed : int
            Random seed for reproducible task generation
        entity_task_ratio : float
            Probability of selecting an entity (parent) table as the focal table.
            Default 0.75 means ~75% of tasks use entity tables as focal.
        relbench_mode : bool
            If True, use RelBench-style tasks: entity as focal AND target,
            DIRECT_ATTRIBUTE_PREDICTION with child-aggregated features.
            Overrides entity_task_ratio to 1.0 when True.
        """
        self.rdb = rdb
        self.random_seed = random_seed
        self.entity_task_ratio = 1.0 if relbench_mode else entity_task_ratio
        self.relbench_mode = relbench_mode
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Initialize data generator
        self.data_generator = TaskDataGenerator(rdb=rdb, random_seed=random_seed)

    def _classify_tables(self) -> Tuple[List[str], List[str]]:
        """Classify tables into entity (has children) and non-entity (no children).

        An entity table is referenced by at least one other table's FK.
        """
        referenced = set()
        for rel in self.rdb.relationships:
            referenced.add(rel.to_table)

        entity_tables = []
        non_entity_tables = []
        for table_name in self.rdb.tables:
            if table_name in referenced:
                entity_tables.append(table_name)
            else:
                non_entity_tables.append(table_name)
        return entity_tables, non_entity_tables

    def _has_children(self, table_name: str) -> bool:
        """Check if a table is referenced by any FK from another table."""
        for rel in self.rdb.relationships:
            if rel.to_table == table_name:
                return True
        return False

    def _get_children(self, table_name: str) -> List[str]:
        """Get tables that reference this table via FK."""
        children = []
        for rel in self.rdb.relationships:
            if rel.to_table == table_name:
                children.append(rel.from_table)
        return children

    def generate_single_table_prediction_tasks(
        self,
        table_name: str,
        table,
        num_tasks: int = 1,
        exclude_primary_foreign_keys: bool = True,
        exclude_timestamp_columns: bool = True,
    ) -> List[Task]:
        """
        Generate single table prediction tasks for a table.

        Parameters
        ----------
        table_name : str
            Name of the table
        table : Table
            Table object
        num_tasks : int
            Number of tasks to generate
        exclude_primary_foreign_keys : bool
            Whether to exclude primary and foreign key columns as targets
        exclude_timestamp_columns : bool
            Whether to exclude timestamp columns as targets

        Returns
        -------
        List[Task]
            List of generated tasks
        """
        tasks = []

        # Find candidate target columns
        candidate_columns = []
        for i, (col_name, data_type_config) in enumerate(
            zip(table.column_names, table.data_type_configs)
        ):
            if exclude_primary_foreign_keys:
                if data_type_config.data_type in [
                    DataType.PRIMARY_KEY,
                    DataType.FOREIGN_KEY,
                ]:
                    continue
            if exclude_timestamp_columns:
                if data_type_config.data_type == DataType.TIMESTAMP:
                    continue
            candidate_columns.append((i, col_name, data_type_config))

        if len(candidate_columns) == 0:
            print(f"Warning: No candidate columns found for table {table_name}")
            return tasks

        # Generate tasks by randomly selecting target columns
        for _ in range(min(num_tasks, len(candidate_columns))):
            # Select a random target column (without replacement)
            selected_idx = random.randint(0, len(candidate_columns) - 1)
            target_col_idx, target_col_name, target_data_type_config = (
                candidate_columns.pop(selected_idx)
            )

            # Determine task type and evaluation metric based on data type
            if target_data_type_config.data_type == DataType.CATEGORICAL:
                dbb_task_type = DBBTaskType.classification
                evaluation_metric = DBBTaskEvalMetric.accuracy
                num_classes = target_data_type_config.config.get("num_categories", 2)
            elif target_data_type_config.data_type == DataType.FLOAT:
                dbb_task_type = DBBTaskType.regression
                evaluation_metric = DBBTaskEvalMetric.mse
                num_classes = None
            else:
                # Not a target column
                continue

            time_column_name = None
            if table.time_column is not None:
                time_column_name = table.column_names[table.time_column]

            # Create task with key table concept
            task_name = f"{table_name}_{target_col_name}_task"
            task = Task(
                task_name=task_name,
                task_type=TaskType.SINGLE_TABLE_PREDICTION,
                key_tables=table_name,  # Single key table for single table prediction
                target_table=table_name,
                target_column=target_col_name,
                real_name_for_target_table=table_name,
                real_name_for_target_column=target_col_name,
                evaluation_metric=evaluation_metric,
                dbb_task_type=dbb_task_type,
                num_classes=num_classes,
                description=f"Predict {target_col_name} column in {table_name} table",
                time_column_name=time_column_name,
            )

            tasks.append(task)

        return tasks

    def generate_tasks_for_rdb_per_table(
        self,
        rdb,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> List[Task]:
        """
        Generate tasks for each table in an RDB.

        Parameters
        ----------
        rdb : RDB
            RDB object
        exclude_small_tables : bool
            Whether to exclude small tables
        min_table_size : int
            Minimum table size to consider for task generation

        Returns
        -------
        List[Task]
            List of generated tasks
        """

        if not rdb.tables:
            return []

        all_tasks = []

        for table_name, table in rdb.tables.items():
            if exclude_small_tables and table.num_rows < min_table_size:
                continue
            tasks = self.generate_single_table_prediction_tasks(table_name, table)
            if len(tasks) > 0:
                all_tasks.extend(tasks)
                print(f"Generated {len(tasks)} tasks for table {table_name}")

        return all_tasks

    def generate_tasks_for_rdb(
        self,
        rdb,
        tasks_per_rdb: int = 3,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> List[Task]:
        """
        Generate tasks for all tables in an RDB.

        Parameters
        ----------
        rdb : RDB
            RDB object
        tasks_per_table : int
            Number of tasks to generate per table
        exclude_small_tables : bool
            Whether to exclude small tables
        min_table_size : int
            Minimum table size to consider for task generation

        Returns
        -------
        List[Task]
            List of generated tasks
        """

        if not rdb.tables:
            return []

        all_tasks = []

        for i in range(tasks_per_rdb):
            print(f"Generating {i+1}th tasks for RDB...")
            table_name, table = random.choice(list(rdb.tables.items()))
            # Skip small tables if requested
            if exclude_small_tables and table.num_rows < min_table_size:
                print(f"Skipping table {table_name} (only {table.num_rows} rows)")
                continue

            # Generate single table prediction tasks for this table
            tasks = self.generate_single_table_prediction_tasks(
                table_name, table, num_tasks=1
            )

            if len(tasks) > 0:
                all_tasks.extend(tasks)
                print(f"Generated {len(tasks)} tasks for table {table_name}")

        return all_tasks

    def generate_tasks_for_rdb_with_complex_tasks(
        self,
        rdb,
        tasks_per_rdb: int = 3,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> List[Task]:
        """Generate tasks for an RDB with complex tasks.

        Args:
            rdb (_type_): RDB object
            tasks_per_rdb (int, optional): Number of tasks to generate per RDB. Defaults to 3.
            exclude_small_tables (bool, optional): Whether to exclude small tables. Defaults to True.
            min_table_size (int, optional): Minimum table size to consider for task generation. Defaults to 10.

        Returns:
            List[Task]: List of generated tasks
        """

        all_tasks = []

        for i in range(tasks_per_rdb):
            print(f"Generating {i+1}th tasks for RDB...")
            focal_table_name, schema_graph = (
                self.generate_random_focal_table_and_schema(
                    rdb=rdb,
                    max_neighbors=2,
                    exclude_small_tables=exclude_small_tables,
                    min_table_size=min_table_size,
                )
            )
            if focal_table_name is None:
                continue
            if len(schema_graph.nodes) < 2:
                print(f"Skipping schema graph with {len(schema_graph.nodes)} nodes")
                continue
            target_table_name = schema_graph.generate_target_table_name(
                root_p=1.0 if self.relbench_mode else 0.0,
            )
            # RelBench mode: entity table is target, predict its own column
            is_entity_target = (
                self.relbench_mode
                and target_table_name == list(schema_graph.nodes.keys())[0]
            )
            task_type = schema_graph.compute_possible_task_types(target_table_name)
            # DIRECT_ATTRIBUTE_PREDICTION (root target) only needs 2 nodes;
            # RELATIONAL_AGGREGATION_PREDICTION still requires 3 for multi-hop joins
            if task_type != TaskType.DIRECT_ATTRIBUTE_PREDICTION and len(schema_graph.nodes) < 3:
                print(f"Skipping schema graph with {len(schema_graph.nodes)} nodes (needs 3 for RELATIONAL_AGGREGATION_PREDICTION)")
                continue
            if task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION:
                if is_entity_target:
                    target_column_name = random.choice(
                        rdb.tables[target_table_name].get_feature_columns(
                            only_categorical=True
                        )
                    )
                else:
                    target_column_name = random.choice(
                        rdb.tables[target_table_name].get_feature_columns(
                            only_categorical=True
                        )
                    )
            elif task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION:
                target_column_name = random.choice(
                    rdb.tables[target_table_name].get_feature_columns(only_float=True)
                )
            else:
                raise ValueError(f"Invalid task type: {task_type}")
            # print(f"Selected target column: {target_column_name}")
            # TODO: The filter condition is not implemented yet
            target_node_set = TargetNodeSet(
                target_table_name, target_column_name, filter_condition=None
            )
            target_computation = None
            if task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION:
                target_computation = DirectAttributeTarget(
                    target_table_name, target_column_name
                )
            elif task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION:
                target_computation = RelationalAggregationTarget(
                    target_node_set,
                    target_column_name,
                    random.choice(AggregationFunctionList),
                    random.choice(PredicateFunctionList),
                )

            # Determine evaluation metric and DBB task type based on target computation
            if isinstance(target_computation, DirectAttributeTarget):
                # For direct attribute, we need to check the column type
                target_table = rdb.tables[target_table_name]
                target_col_idx = target_table.column_names.index(target_column_name)
                target_data_type = target_table.data_type_configs[
                    target_col_idx
                ].data_type

                if target_data_type == DataType.CATEGORICAL:
                    evaluation_metric = DBBTaskEvalMetric.accuracy
                    dbb_task_type = DBBTaskType.classification
                    num_classes = target_table.data_type_configs[
                        target_col_idx
                    ].config.get("num_categories", 2)
                else:
                    evaluation_metric = DBBTaskEvalMetric.mse
                    dbb_task_type = DBBTaskType.regression
                    num_classes = None
            else:
                # For relational aggregation, it's always binary classification
                evaluation_metric = DBBTaskEvalMetric.accuracy
                dbb_task_type = DBBTaskType.classification
                num_classes = 2

            task = Task(
                task_name=f"complex_task_{focal_table_name}_{task_type.value}_{i+1}",
                task_type=task_type,
                key_tables=[focal_table_name],
                target_table=target_table_name,
                real_name_for_target_table=focal_table_name,
                target_column=target_column_name,
                real_name_for_target_column=(
                    target_column_name
                    if task_type == TaskType.SINGLE_TABLE_PREDICTION
                    else "labels"
                ),
                evaluation_metric=evaluation_metric,
                dbb_task_type=dbb_task_type,
                num_classes=num_classes,
                target_node_set=target_node_set,
                target_computation=target_computation,
                schema_graph=schema_graph,
                description=f"Complex task: {task_type.value} on {target_table_name}.{target_column_name}",
            )
            all_tasks.append(task)

        return all_tasks

    def generate_task_data(
        self,
        tasks: List[Task],
        rdb,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> None:
        """
        Generate complete data for all tasks via TaskDataGenerator.generate_task_data.

        Parameters
        ----------
        tasks : List[Task]
            List of tasks to generate data for
        rdb : RDB
            RDB object containing the tables
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation
        """
        for task in tasks:
            task_data = self.data_generator.generate_task_data(
                task=task,
                key_tables=rdb.tables,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
            )
            if task_data is None:
                print(f"Warning: Task {task.task_name} has no data, skipping")
                continue
            task.set_task_data(task_data)

            # Create and store task generation schema
            primary_table_name = task.get_primary_key_table()
            if primary_table_name in rdb.table_generation_schemas:
                primary_table_schema = rdb.table_generation_schemas[primary_table_name]

                task_schema = TaskGenerationSchema(
                    task_name=task.task_name,
                    primary_table_name=primary_table_name,
                    primary_table_generation_schema=primary_table_schema,
                    target_column=task.target_column,
                    task_type=task.task_type.value,
                )
                rdb.task_generation_schemas.append(task_schema)

            print(f"Generated data for task: {task.task_name}")

    def save_all_task_data(
        self,
        tasks: List[Task],
        base_path: str,
    ) -> List[DBBTaskMeta]:
        """
        Save all task data to files and return task metadata.

        Parameters
        ----------
        tasks : List[Task]
            List of tasks with generated data
        base_path : str
            Base directory to save task data

        Returns
        -------
        List[DBBTaskMeta]
            List of task metadata objects
        """
        task_metas = []

        for task in tasks:
            if task.task_data is None:
                print(f"Warning: Task {task.task_name} has no data, skipping")
                continue

            # Create task directory
            task_file_dir = os.path.join(base_path, task.task_name)
            os.makedirs(task_file_dir, exist_ok=True)

            # Save task data
            self.data_generator.save_task_data(task.task_data, task_file_dir)

            # Create task metadata
            source_file = f"{task.task_name}/{{split}}.parquet"
            task_meta = task.to_dbb_task_meta(source_file)
            task_metas.append(task_meta)

        print(f"Saved {len(task_metas)} tasks to {base_path}")
        return task_metas

    def generate_random_focal_table_and_schema(
        self,
        rdb,
        max_neighbors: int = 2,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> Tuple[str, SchemaGraph]:
        """
        Randomly select a focal entity table and sample neighbors to create a schema graph.

        Biased toward entity (parent) tables: entity_task_ratio of the time, selects
        a table that has children, producing RelBench-style aggregation tasks.

        Parameters
        ----------
        rdb : RDB
            The RDB object containing tables and relationships
        max_neighbors : int
            Maximum number of neighbor tables to select (default: 2)
        exclude_small_tables : bool
            Whether to exclude small tables from selection (default: True)
        min_table_size : int
            Minimum table size to consider for focal entity selection (default: 10)

        Returns
        -------
        Tuple[str, SchemaGraph]
            A tuple containing:
            - Name of the randomly selected focal table
            - Schema graph containing the focal table and its selected neighbors
        """
        entity_tables, non_entity_tables = self._classify_tables()

        # Filter by size
        if exclude_small_tables:
            entity_tables = [
                t for t in entity_tables
                if rdb.tables[t].num_rows >= min_table_size
            ]
            non_entity_tables = [
                t for t in non_entity_tables
                if rdb.tables[t].num_rows >= min_table_size
            ]

        # Weighted sampling: bias toward entity tables
        if entity_tables and (
            random.random() < self.entity_task_ratio or not non_entity_tables
        ):
            focal_table_name = random.choice(entity_tables)
        elif non_entity_tables:
            focal_table_name = random.choice(non_entity_tables)
        else:
            print(
                f"Warning: No suitable tables found for focal entity selection "
                f"(min_size={min_table_size}, exclude_small={exclude_small_tables}). "
                f"All {len(rdb.tables)} tables have < {min_table_size} rows. Skipping."
            )
            return None, None

        # Build schema graph: when focal is entity, bias first-hop neighbors toward children
        schema_graph = rdb.create_multi_hop_schema_graph_biased(
            focal_table_name,
            num_hops=2,
            neighbors_each_hop=1,
            prefer_children=(focal_table_name in entity_tables),
            get_children_fn=self._get_children,
        )

        return focal_table_name, schema_graph
