# Released data

The interaction files and constructed item/user knowledge graphs are available in the [review-release archive](https://anonymous.4open.science/api/repo/MKGE-3556/zip). Download the archive and extract its `data/` directory into the repository root. This uses the current released data, including the updated Beauty item graph.

```bash
curl -L https://anonymous.4open.science/api/repo/MKGE-3556/zip -o /tmp/MKGE-review-release.zip
unzip /tmp/MKGE-review-release.zip 'data/*' -d .
```

Expected paths:

| Dataset | Interaction file | Item graph | User graph |
| --- | --- | --- | --- |
| MovieLens 1M | `data/ml-1m/ml-1m.inter` | `data/ml-1m/ml-1m.item.kg` | `data/ml-1m/ml-1m.user.kg` |
| Amazon Beauty | `data/amazon-beauty/amazon-beauty.inter` | `data/amazon-beauty/amazon-beauty.item.kg` | `data/amazon-beauty/amazon-beauty.user.kg` |
| Amazon Video Games | `data/amazon-videogames/amazon-videogames.inter` | `data/amazon-videogames/amazon-videogames.item.kg` | `data/amazon-videogames/amazon-videogames.user.kg` |

Then follow the dataset-specific training commands in the root README.
