# Contribuindo com o GolData

## Setup

```bash
git clone https://github.com/fabriciojunio/goldata.git
cd goldata
pip install -e ".[dev]"
```

## Rodando os Testes

```bash
pytest tests/ -v --cov=goldata --cov-report=html
```

## Adicionando um Novo Modelo

1. Crie o arquivo em `goldata/models/<categoria>/`
2. Herde de `goldata.models.base.BaseModel`
3. Implemente `fit()` e `predict()`
4. Adicione testes em `tests/unit/models/`
5. Registre na documentação `docs/modelos.md`

## Padrão de Commits

```
feat(xg): melhorar features com dados de pressão defensiva
fix(api): corrigir paginação nos endpoints de métricas
perf(db): adicionar índice na tabela de partidas
test(monte-carlo): aumentar cobertura para 95%
```
