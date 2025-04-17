// Research Prospectus Template
// A template for 1-page research prospectus documents

#let prospectus(
  title: "Research Prospectus",
  authors: (
    (name: "Author Name", affiliations: (1,), email: "email@institution.edu"),
  ),
  affiliations: (
    (number: 1, name: "Institution Name"),
  ),
  program: "Program Name",
  date: "Current Date",
  bibfile: none,
  body
) = {
  // Page setup for compact 1-page prospectus
  set page(
    paper: "us-letter",
    margin: (x: 1.8cm, y: 1.5cm),
  )

  // Font setup
  set text(
    font: "New Computer Modern",
    size: 10pt,
    hyphenate: false,
  )

  // Compact title and header with proper author affiliations
  align(center)[
  #block(
    width: 100%,  // Adjust percentage to control width
    text(weight: "bold", size: 13pt)[#title]
  )

    #block(text(style: "italic", size: 9pt)[Research prospectus in response to FOA #program · #date])

    // Authors with superscript affiliations and emails
    #block(
      {
        let author_elements = ()
        for author in authors {
          let name = author.name

          // Add superscript affiliations
          let affs = ""
          if "affiliations" in author {
            // Check if affiliations is an array or a single number
            let aff_list = if type(author.affiliations) == array {
              author.affiliations
            } else {
              (author.affiliations,)  // Convert to single-item array
            }

            if aff_list.len() > 0 {
              // Add a thin space before the superscript
              affs = [#h(0.2em)] + super([#aff_list.map(a => str(a)).join(",")])
            }
          }

          // Create author element with affiliations
          author_elements.push([#name#affs])
        }
        author_elements.join([#h(0.5em)·#h(0.5em)])
      }
    )

    // Affiliations
    #v(0.1em)
    #block(
      text(size: 8.5pt)[
        #affiliations.map(aff => {
          // Add a thin space between the superscript and the institution name
          [#super([#str(aff.number)])#h(0.2em)#aff.name]
        }).join([#h(0.7em)·#h(0.7em)])
      ]
    )

    // Emails - display each author's email on same line
    #v(0.1em)
    #block(
      text(size: 8.5pt)[
        #authors.map(author => {
          if "email" in author {
            author.email
          } else {
            ""
          }
        }).filter(email => email != "").join([#h(0.5em)·#h(0.5em)])
      ]
    )

    #v(0.1em)
    #line(length: 30%, stroke: 0.5pt)
  ]

  // Main content formatting with improved heading display
  set heading(
    numbering: none,
    outlined: false,
  )

  show heading.where(level: 1): it => {
    v(0.5em)
    text(weight: "bold", size: 11pt)[#it.body]
    v(-0.3em)
  }

  set par(
    justify: true,
    leading: 0.65em,
  )

  // Body content
  body

  // Add bibliography if bibfile is provided
  if bibfile != none {
    // Bibliography settings
    show bibliography: set text(8pt)
    bibliography(bibfile, style: "american-psychological-association")
  }
}