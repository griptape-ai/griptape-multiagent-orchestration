from __future__ import annotations

import schema
import yaml
from attrs import define
from yaml.resolver import Resolver

from griptape_statemachine.parsers.base_parser import BaseParser

STRUCTURE_SCHEMA = schema.Schema(
    {
        schema.Optional("model"): str,
        schema.Optional("ruleset_ids"): [str],
    }
)

CONFIG_SCHEMA = schema.Schema(
    {
        "rulesets": schema.Schema(
            {
                str: schema.Schema(
                    {
                        "name": str,
                        "rules": [str],
                    }
                )
            }
        ),
        "assistants": schema.Schema({str: STRUCTURE_SCHEMA}),
        "events": schema.Schema(
            {
                str: schema.Schema(
                    {
                        "transitions": [
                            schema.Schema(
                                {
                                    "from": str,
                                    "to": str,
                                    schema.Optional("internal"): bool,
                                    schema.Optional("on"): str,
                                }
                            )
                        ],
                    }
                )
            }
        ),
        "states": schema.Schema(
            {
                str: schema.Schema(
                    {
                        schema.Optional(
                            schema.Or("initial", "final")
                        ): bool,  # pyright: ignore[reportArgumentType]
                        schema.Optional("assistants"): schema.Schema(
                            {str: STRUCTURE_SCHEMA}
                        ),
                    }
                )
            }
        ),
    }
)


@define()
class ConfigParser(BaseParser):
    def __attrs_post_init__(self) -> None:
        # remove resolver entries for On/Off/Yes/No
        # https://stackoverflow.com/questions/36463531/pyyaml-automatically-converting-certain-keys-to-boolean-values
        for ch in "OoYyNn":
            if ch in Resolver.yaml_implicit_resolvers:
                if len(Resolver.yaml_implicit_resolvers[ch]) == 1:
                    del Resolver.yaml_implicit_resolvers[ch]
                else:
                    Resolver.yaml_implicit_resolvers[ch] = [
                        x
                        for x in Resolver.yaml_implicit_resolvers[ch]
                        if x[0] != "tag:yaml.org,2002:bool"
                    ]

    def parse(self) -> dict:
        data = yaml.safe_load(self.file_path.read_text())

        CONFIG_SCHEMA.validate(data)

        return data
